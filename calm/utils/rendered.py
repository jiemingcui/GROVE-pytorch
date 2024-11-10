import torch
import torch.nn as nn

class MLP(nn.Module):
    # TODO ask jiale to update
    def __init__(self, input_dim, hidden_dims, output_dim): # 73, 256, 1024, 512
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

class MotionImgEvaluator():
    def __init__(self):
        self.device = "cuda:0"
        # self.motion_encoder = MLP(73, [256, 1024], 512)

        model_dir = "/home/cjm/CALM_related/anyskill/rendered/blender_model/calm.pth"
        self.motion_encoder = torch.load(model_dir, map_location=self.device)
        # checkpoint.eval()
        # self.motion_encoder.load_state_dict(checkpoints[])
        # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoints['epoch']))

        self.motion_encoder.to(self.device)

        self.motion_encoder.eval()

    def get_motion_embedding(self, motion):
        with torch.no_grad():
            m_lens = torch.ones(motion.shape[0])
            motion = motion.detach().to(self.device).float()  # [1,16,13] [1024,17,13]
            motion_embedding = self.motion_encoder(motion)  # [1,4,512]
        return motion_embedding


def test():
    evaluator = MotionImgEvaluator()
    return evaluator
