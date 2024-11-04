import torch
import torch.nn as nn
import os
import mlflow
import logging
import time
import yaml
import uuid
from torch.utils.tensorboard.writer import SummaryWriter
from typing import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR


from intensfusion.models.mm_container import Unimodal
from intensfusion.utils.visualization import Visualizer
from intensfusion.eval_scripts.performance import \
    AUPRC, f1_score, accuracy, eval_affect
from intensfusion.eval_scripts.complexity import \
    all_in_one_train, all_in_one_test

softmax = nn.Softmax()


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with
    different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size()) - 1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) in [
        nn.MSELoss,
        nn.modules.loss.BCEWithLogitsLoss,
        nn.L1Loss,
        nn.SmoothL1Loss,
        nn.HuberLoss,
    ]:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)


def train(
    encoder,
    head,
    train_dataloader,
    valid_dataloader,
    total_epochs,
    additional_optimizing_modules=[],
    is_packed=False,
    early_stop=False,
    task="classification",
    optimtype: type[torch.optim.Optimizer] = torch.optim.SGD,
    lr=0.001,
    weight_decay=0.0,
    objective=nn.CrossEntropyLoss(),
    auprc=False,
    save="best.pt",
    validtime=False,
    objective_args_dict=None,
    input_to_float=True,
    clip_val=8,
    track_complexity=True,
    momentum: float = 0.9,
    save_path="./",
    scheduler_step_size=30,
    scheduler_gamma=0.1,
    writer: SummaryWriter | None = None,
    modality_idx=0
):
    """
    Handle running a simple supervised training loop.

    :param encoders: list of modules, unimodal encoders for each input
                    modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list
                    and outputs fused representation
    :param head: classification or prediction head,
                takes in output of fusion module and outputs
                the classification or prediction results that will be sent
                to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules
            that you want to be optimized by the optimizer other than those in
            encoders,fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or
            not (default is False, which means we expect input of
            [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)]
            for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not
            improve over 7 epochs
    :param task: type of task, currently support "classification",
            "regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of
            CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom
            objective function that takes in three arguments: prediction,
            ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with
            current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into
            objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    :param reg_param(float): only applicable for MKL fusion.Penalty
            for the Lp norm regularizer
    :paran p(float): only applicable for MKL fusion.
            defines p value of the Lp norm
    :param momentum(float): momentum for SGD with momoentum
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    objective = objective.to(device)
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.makedirs(os.path.join(save_path, "checkpoints"))

    print("Settimg up training with a single modality")
    model = Unimodal(
        encoder=encoder, head=head, has_padding=is_packed
    ).to(device)

    def _save_hparams():
        train_config = OrderedDict(
            {
                "head": head.__class__.__name__,
                "lr": lr,
                "weight_decay": weight_decay,
                "task": task,
                "encoder": type(encoder),
                "epochs_trained": total_epochs,
                "optimizer": str(optimtype),
                "objective": str(objective),
                "scheduler_step_size": scheduler_step_size,
                "scheduler_gamma": scheduler_gamma,
                "uuid": str(uuid.uuid4()),
            }
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(os.path.join(save_path, "checkpoints"))
        with open(os.path.join(save_path, "train_hparams.yaml"), "w") as file:
            yaml.dump(train_config, file)

    def _optim_param_list(model, add_params):
        assert type(model).__name__ == "Unimodal"
        return [
            {"params": model.encoder.parameters(),
             "weight_decay": weight_decay},
            {"params": model.head.parameters(),
             "weight_decay": weight_decay},
            {"params": add_params,
             "weight_decay": weight_decay}
        ]

    def _trainprocess():
        print("Starting Training..")
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend([p for p in m.parameters()
                                        if p.requires_grad])
        optim_param_list = _optim_param_list(model, additional_params)
        if "SGD" in str(optimtype):
            op = optimtype(optim_param_list, lr=lr, momentum=momentum)
        else:
            op = optimtype(optim_param_list, lr=lr)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op,patience=scheduler_step_size, mode='min', factor=scheduler_gamma)
        # scheduler =  StepLR(op, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scheduler = ExponentialLR(op, gamma=scheduler_gamma)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    # with torch.backends.cudnn.flags(enabled=False):
                    assert False
                    model.train()
                    out = model([[_processinput(i).to(device) for i in j[0]], j[1]])

                else:
                    model.train()
                    features = j[0][modality_idx]
                    out = model(features.to(device, non_blocking=True))
                if not (objective_args_dict is None):
                    objective_args_dict["reps"] = model.reps
                    objective_args_dict["fused"] = model.fuseout
                    objective_args_dict["inputs"] = j[:-1]
                    objective_args_dict["training"] = True
                    objective_args_dict["model"] = model
                loss = deal_with_objective(objective,
                                           out,
                                           j[-1],
                                           objective_args_dict)
                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            train_loss_ = totalloss / totals
            print(f"Epoch: {str(epoch)} train loss: {str(train_loss_.item())}")
            try:
                writer.add_scalar("Loss/train", train_loss_.item(), epoch)
                mlflow.log_metric("train_loss", train_loss_.item(), step=epoch)
            except AttributeError as aerr:
                print(f"Logging modules not passed: {aerr}")

            validstarttime = time.time()
            if validtime:
                print("train total: " + str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        assert False
                        model.eval()
                        out = model([[_processinput(i).to(device) for i in j[0]], j[1]])
                    else:
                        model.eval()
                        features = j[0][modality_idx]
                        out = model(features.to(device, non_blocking=True))

                    if not (objective_args_dict is None):
                        objective_args_dict["reps"] = model.reps
                        objective_args_dict["fused"] = model.fuseout
                        objective_args_dict["inputs"] = j[:-1]
                        objective_args_dict["training"] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict
                    )
                    totalloss += loss * len(j[-1])

                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        sm = softmax(out)
                        pts += [
                            (sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))
                        ]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss / totals
            scheduler.step()
            if task == "classification":
                val_acc = accuracy(true, pred)
                print(
                    "Epoch "
                    + str(epoch)
                    + " valid loss: "
                    + str(valloss.item())
                    + " acc: "
                    + str(val_acc)
                )
                writer.add_scalar("Loss/val", valloss.item(), epoch)
                mlflow.log_metric("val_loss", valloss.item(), step=epoch)
                if val_acc > bestacc:
                    patience = 0
                    bestacc = val_acc
                    print("Saving Best Val Acc")
                    torch.save(
                        model, os.path.join(save_path, "checkpoints",
                                            f"{save}_best_acc.pt")
                    )
                else:
                    patience += 1
                torch.save(
                    model, os.path.join(save_path, "checkpoints", f"{save}_latest.pt")
                )
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best Val Loss")
                    # torch.save(model, save)
                    # TODO: Save the Dict instead of the model
                    torch.save(
                        model, os.path.join(save_path, "checkpoints", f"{save}_best.pt")
                    )
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print(
                    "Epoch "
                    + str(epoch)
                    + " valid loss: "
                    + str(valloss)
                    + " f1_micro: "
                    + str(f1_micro)
                    + " f1_macro: "
                    + str(f1_macro)
                )
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch " + str(epoch) + " valid loss: " + str(valloss.item()))
                try:
                    writer.add_scalar("Loss/val", valloss.item(), epoch)
                    mlflow.log_metric("val_loss", valloss.item(), step=epoch)
                except AttributeError:
                    print("logging modules not pased")
                except:
                    print("logging error")
                train_accs = test(
                    model=model,
                    test_dataloaders_all=train_dataloader,
                    is_packed=True,
                    criterion=torch.nn.L1Loss(),
                    task="posneg-classification",
                    no_robust=True,
                )
                val_accs = test(
                    model=model,
                    test_dataloaders_all=valid_dataloader,
                    is_packed=True,
                    criterion=torch.nn.L1Loss(),
                    task="posneg-classification",
                    no_robust=True,
                )

                mlflow.log_metrics(train_accs, step=epoch)
                mlflow.log_metrics(
                    {
                        "Val_Accuracy": val_accs["Accuracy"],
                        "Val_Acc_include_zero": val_accs["Acc_include_zero"],
                    },
                    step=epoch,
                )
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    # torch.save(model, save)
                    # TODO: Save the Dict instead of the model
                    # TODO: Save the model with best Acc as well, 
                    # as that is what we are using for the final test
                    torch.save(
                        model, os.path.join(save_path, "checkpoints", f"{save}_best.pt")
                    )
                else:
                    patience += 1
                if val_accs["Accuracy"] > bestacc:
                    bestacc = val_accs["Accuracy"]
                    torch.save(
                        model,
                        os.path.join(save_path, "checkpoints", f"{save}_best_acc.pt"),
                    )
                torch.save(
                    model, os.path.join(save_path, "checkpoints", f"{save}_latest.pt")
                )
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: " + str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  " + str(validendtime - validstarttime))
                print("Valid total: " + str(totals))

    _save_hparams()
    if track_complexity:
        all_in_one_train(_trainprocess, [model] +
                         additional_optimizing_modules)
    else:
        _trainprocess()
    try:
        writer.flush()
    except:
        print("logging error")
    return model


def single_test(
    model,
    test_dataloader,
    modality_index,
    is_packed=False,
    criterion=nn.CrossEntropyLoss(),
    task="classification",
    auprc=False,
    input_to_float=True,
    save_path="./"
):
    """Run single test for model.

    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp

    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                assert False
                out = model(
                    [
                        [
                            _processinput(i).to(
                                torch.device(
                                    "cuda:0" if torch.cuda.is_available() else "cpu"
                                )
                            )
                            for i in j[0]
                        ],
                        j[1],
                    ]
                )
            else:
                # out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                #             for i in j[:-1]])
                features = j[0][modality_index]
                out = model(features.to(device, non_blocking=True))
            if (
                type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss
                or type(criterion) == torch.nn.MSELoss
            ):
                loss = criterion(
                    out,
                    j[-1].float().to(device),
                )

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size()) - 1)
                else:
                    truth1 = j[-1]
                loss = criterion(
                    out,
                    truth1.long().to(device),
                )
            else:
                loss = criterion(out, j[-1].to(device))
            totalloss += loss * len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [
                    (sm[i][1].item(), j[-1][i].item()) for i in range(j[-1].size(0))
                ]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss / totals
        if auprc:
            print("AUPRC: " + str(AUPRC(pts)))
        if task == "classification":
            print("acc: " + str(accuracy(true, pred)))
            return {"Accuracy": accuracy(true, pred)}
        elif task == "multilabel":
            print(
                " f1_micro: "
                + str(f1_score(true, pred, average="micro"))
                + " f1_macro: "
                + str(f1_score(true, pred, average="macro"))
            )
            return {
                "micro": f1_score(true, pred, average="micro"),
                "macro": f1_score(true, pred, average="macro"),
            }
        elif task == "regression":
            print("mse: " + str(testloss.item()))
            return {"MSE": testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            # print("acc: "+str(accs) + ', ' + str(acc2))
            return {"Accuracy": accs, "Acc_include_zero": acc2}


def test(
    model,
    test_dataloaders_all,
    modality_index,
    dataset="default",
    method_name="My method",
    is_packed=False,
    criterion=nn.CrossEntropyLoss(),
    task="classification",
    auprc=False,
    input_to_float=True,
    no_robust=False,
):
    """
    Handle getting test results for a simple supervised training loop.

    :param model: saved checkpoint filename from train
    :param test_dataloaders_all: test data
    :param dataset: the name of dataset, need to be set for testing effective robustness
    :param criterion: only needed for regression, put MSELoss there
    """
    if no_robust:

        def _testprocess():
            single_test_result = single_test(
                model=model,
                test_dataloader=test_dataloaders_all,
                modality_index=modality_index,
                is_packed=is_packed,
                criterion=criterion,
                task=task,
                auprc=auprc,
                input_to_float=input_to_float
            )
            return single_test_result

        test_result = all_in_one_test(_testprocess, [model])
        return test_result
