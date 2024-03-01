import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from models.jepa import JEPA
from utils.misc import JEPAParams


def test_jepa():

    encoder_x = nn.Linear(128, 128)
    encoder_y = copy.deepcopy(encoder_x)
    predictor = nn.Linear(128, 128)
    hparams = JEPAParams()

    model = JEPA(encoder_x, encoder_y, predictor, hparams, training=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)

    # test data and training
    x = torch.randn(128, 128)
    y = torch.ones(128, 128)

    for epoch in range(200):
        optimizer.zero_grad()
        pred_y, embed_y, losses = model(x, y)
        loss = criterion(pred_y, embed_y) + losses['loss']
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    print('pred_y: ', pred_y)
    print('embed_y: ', embed_y)


if __name__ == "__main__":
    test_jepa()
