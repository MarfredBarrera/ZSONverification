"""HM3D semantic.glb -> per-instance vertices (CPU, numpy only)."""
import json, struct
import numpy as np


def objects(scene, DATA):
    """HM3D semantic.glb -> {object id: (category, vertices Nx3)}; objects are matched to semantic.txt by vertex colour."""
    base = f"{DATA}/scene_datasets/hm3d/val/{scene}/{scene.split('-')[1]}.semantic"
    col = {}
    for l in open(base + ".txt").read().splitlines()[1:]:
        i, c, n, _ = l.split(",", 3); col[int(c, 16)] = (int(i), n.strip('"'))
    b = open(base + ".glb", "rb").read(); n = struct.unpack("<I", b[12:16])[0]; j = json.loads(b[20:20 + n]); bin0 = 20 + n + 8
    def acc(k):
        a = j["accessors"][k]; v = j["bufferViews"][a["bufferView"]]
        dt = {5126: "<f4", 5121: "u1", 5123: "<u2"}[a["componentType"]]; w = {"VEC3": 3, "VEC4": 4}[a["type"]]
        o = bin0 + v.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(b, dt, a["count"] * w, o).reshape(-1, w), a.get("normalized", False), a["componentType"]
    P, C = {}, {}
    for m in j["meshes"]:
        for p in m["primitives"]:
            xyz = acc(p["attributes"]["POSITION"])[0]; c, _, ct = acc(p["attributes"]["COLOR_0"])
            c = c[:, :3] / (65535. if ct == 5123 else 1.)  # normalized ushort, linear -> sRGB bytes
            c = np.rint(np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255).astype(int)
            key = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
            for k in np.unique(key):
                if int(k) in col: P.setdefault(col[int(k)], []).append(xyz[key == k])
    return {i: (n, np.concatenate(v)) for (i, n), v in P.items()}
