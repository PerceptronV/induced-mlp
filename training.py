import torch
from tqdm import tqdm

from pruning import no_pruning


def train(
    model,
    n_epochs,
    optimiser,
    train_dataloader,
    train_criterion,
    val_dataloader=None,
    val_criterion=None,
    pruner=no_pruning,
    early_stop=0,
    show_pbar=True,
    leave_pbar=True,
    verbose=True,
    its=None
):
    
    def apply(*args, **kwargs):
        if its is None:
            return model(*args, **kwargs)
        else:
            return model(*args, **kwargs, its=its)

    step = 0
    train_losses = []
    val_losses = []
    val_steps = []

    for epoch in range(n_epochs):

        train_loss = 0
        pbar = tqdm(enumerate(train_dataloader),
                    leave=leave_pbar,
                    disable=not show_pbar)

        for batch, (x, y) in pbar:

            ctx = {
                'model': model,
                'step': step,
                'epoch': epoch,
                'batch': batch,
                'total_steps': len(train_dataloader),
                'total_epochs': n_epochs,
                'progress': epoch / n_epochs,
                'train': True
            }

            optimiser.zero_grad()
            y_pred = apply(x)
            loss = train_criterion(y_pred, y, **ctx)
            loss.backward()
            optimiser.step()

            _loss = loss.item()
            train_losses.append(_loss)
            pbar.set_postfix({'loss': _loss})
            train_loss += _loss
            step += 1

            pruner.prune(model, **ctx)
        
        if val_dataloader is not None:
            val_loss = 0
            with torch.no_grad():
                for x, y in val_dataloader:
                    ctx = {
                        'model': model,
                        'step': step,
                        'epoch': epoch,
                        'batch': batch,
                        'train': False
                    }
                    y_pred = apply(x, weights=model.weights.triu())
                    loss = val_criterion(y_pred, y, **ctx)
                    _loss = loss.item()
                    val_loss += _loss
                    val_losses.append(_loss)
                    val_steps.append(step-1)

        train_loss = train_loss / len(train_dataloader)
        description = f'EP {epoch:4d} Trn Loss: {train_loss:.4f}'
        if val_dataloader is not None:
            val_loss = val_loss / len(val_dataloader)
            description += f' Val Loss: {val_loss:.4f}'
        pbar.set_description(description)
        
        if early_stop is not None and train_loss < early_stop:
            if verbose:
                print(f'Early stop. {description}')
            break
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_steps': val_steps
    }
