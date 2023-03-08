from model_training_utils import add_hparams, EarlyStopping
from get_device_config import *
from loss_function import *
from set_optmizer_and_scheduler import *
from set_summary_writer import *
from model_utils import *
from torch_metrics_utils import *
from tqdm.auto import tqdm


# create some aliases
loss, opt, sched = loss_fn, optimizer, scheduler

checkpoint_path = os.path.join(tensorboard_dir, "model_ckpt_{epoch}.pt")
early_stopping = EarlyStopping(
    patience=10,
    path=checkpoint_path,
    min_epochs=30,
)

progress_bar = tqdm(range(num_epochs), desc="Training: ", unit="epoch(s)")
for N in progress_bar:
    train_loss, val_loss, train_m, val_m = run_epoch(
        N,
        model,
        loss,
        opt,
        sched,
        train_loader,
        val_loader,
        train_metrics,
        val_metrics,
        writer,
    )

    # update progress bar
    train_m_copy = {f"train_{k}".lower(): v.cpu().numpy() for k, v in train_m.items()}
    val_m_copy = {f"val_{k}".lower(): v.cpu().numpy() for k, v in val_m.items()}
    progress_bar.set_postfix(**train_m_copy, **val_m_copy)

    # early stopping when validation loss stops improving
    early_stopping.path = checkpoint_path.format(epoch=N)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(
            f"Early stopping at epoch {N}"
            f" with validation loss {val_loss:.3f}"
            f" and training loss {train_loss:.3f}"
        )
        break

    # TODO (homework): save checkpoint every 10 epochs

# add hyperparameters and corresponding results to tensorboard HParams table
hparam_dict = {
    "backbone": model_name,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "num_classes": num_classes,
    "binary_mask": binary,
    "optimizer": optimizer.__class__.__name__,
    "max_lr": max_lr,
    "loss_function": loss_fn.__class__.__name__,
}
print(train_m)
metrics_dict = {
    "train/end_epoch": N,
    "train/loss": train_loss,
    "train/Accuracy": train_m["MulticlassAccuracy"],
    "val/loss": val_loss,
    "val/Accuracy": val_m["MulticlassAccuracy"],
}
add_hparams(writer, hparam_dict, metrics_dict, epoch_num=N)
writer.close()

# save model to tensorboard folder
model_path = os.path.join(tensorboard_dir, f"model_ckpt_{N+1}.pt")
print(model_path)
torch.save(model.state_dict(), model_path)
