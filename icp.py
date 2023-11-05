import torch
from pytorch3d.io import load_ply, save_obj
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calc_dist(source, target):
    dist_total = 0
    for i in range(source.shape[1]):
        _, dist = closest_point_idx(source[:, i], target)
        dist_total += dist
    return dist_total / source.shape[1]

def closest_point_idx(point, target):
    dist = target - point.unsqueeze(-1)
    dist = torch.sum(dist * dist, dim=0)
    idx = torch.argmin(dist)
    min_dist = torch.sqrt(dist[idx])
    return idx, min_dist

def icp(source, target):
    source = source.T.to(DEVICE)
    target = target.T.to(DEVICE)

    original_source = source.clone().to(DEVICE)

    # mean
    source_mean = torch.tensor([source[0].mean(), source[1].mean(), source[2].mean()]).unsqueeze(-1).to(DEVICE)
    target_mean = torch.tensor([target[0].mean(), target[1].mean(), target[2].mean()]).unsqueeze(-1).to(DEVICE)

    # center
    source = source - source_mean
    target = target - target_mean

    k = 0
    n = source.shape[1]
    R_best = R = torch.eye(3).to(DEVICE)
    t_best = t = torch.zeros(3, 1).to(DEVICE)
    source_to_target = torch.zeros(n).to(DEVICE)
    loss = 1e5
    loss_best = 1e5
    while loss > 1e-3 and k < 30:
        source_new = torch.mm(R, source) + t
        for i in range(n):
            idx, _ = closest_point_idx(source_new[:, i], target)
            source_to_target[i] = idx

        H = torch.zeros((3, 3)).to(DEVICE)
        for i in range(n):
            H += torch.mm(
                source[:, i].unsqueeze(-1),
                target[:, int(source_to_target[i])].unsqueeze(0)
            ).to(DEVICE)

        u, _, v = np.linalg.svd(H.cpu().numpy())
        u = torch.from_numpy(u).to(DEVICE)
        v = torch.from_numpy(v).to(DEVICE)

        weight = torch.eye(3).to(DEVICE)
        weight[2, 2] = torch.det(torch.mm(v, u.T))
        R = torch.mm(torch.mm(v, weight), u.T)
        t = target_mean - torch.mm(R, source_mean)

        source_new = torch.mm(R, source) + t
        loss = calc_dist(source_new, target)
        k += 1

        print("Iteration %d, loss: %.6f" % (k, loss))

        if loss < loss_best:
            loss_best = loss
            R_best = R
            t_best = t

    return torch.mm(R_best, original_source) + t_best

def plot_pointcloud(points, title=""):
    from mpl_toolkits.mplot3d import Axes3D

    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.cpu()
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 220)
    plt.show()

def main():
    source_verts, _ = load_ply("data/source.ply")
    target_verts, _ = load_ply("data/target.ply")

    plot_pointcloud(source_verts.T)
    plot_pointcloud(target_verts.T)

    result = icp(source_verts, target_verts)

    plot_pointcloud(result)



if __name__ == '__main__':
    main()