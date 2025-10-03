# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass
import numpy as np
import trimesh

from isaaclab.terrains.trimesh.mesh_terrains import make_border  # reuse Isaac Lab helper
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshInvertedPyramidStairsTerrainCfg


def inverted_pyramid_stairs_terrain_with_lip(difficulty: float, cfg):
    """
    Like isaaclab.terrains.trimesh.mesh_terrains.inverted_pyramid_stairs_terrain
    but adds optional thin lips along stair edges.

    Expects extra cfg fields:
      - lip_height: float (m)  > 0 enables lips
      - lip_depth:  float (m)  > 0 enables lips
      - lip_inset:  float (m)  +inward from edge, -outward (overhang)
    """
    # Resolve step height via curriculum
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # Number of steps per side (same logic as stock)
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))

    total_height = (num_steps + 1) * step_height
    meshes_list = []

    # Border (unchanged)
    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, step_height, border_center)

    # Terrain bounds
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)

    # Optional lip params (defaults -> off)
    lip_h = getattr(cfg, "lip_height", 0.0)
    lip_d = getattr(cfg, "lip_depth", 0.0)
    lip_inset = getattr(cfg, "lip_inset", 0.0)
    use_lip = (lip_h is not None and lip_h > 0.0) and (lip_d is not None and lip_d > 0.0)

    for k in range(num_steps):
        # Ring size
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)

        # Vertical placement of this ring (same as stock)
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        box_height = total_height - (k + 1) * step_height

        # --- main ring geometry (unchanged) ---
        # top/bottom
        box_dims_tb = (box_size[0], cfg.step_width, box_height)
        # top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        meshes_list.append(trimesh.creation.box(box_dims_tb, trimesh.transformations.translation_matrix(box_pos)))
        # bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        meshes_list.append(trimesh.creation.box(box_dims_tb, trimesh.transformations.translation_matrix(box_pos)))

        # right/left
        if cfg.holes:
            box_dims_rl = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims_rl = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos)))
        # left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims_rl, trimesh.transformations.translation_matrix(box_pos)))

        # ----- lip (true nosing on inner edge, flush with tread, protruding into trench) -----
        if use_lip and not cfg.holes:
            EPS = 2.0e-3  # small lift to avoid z-fighting on the top face

            # Tread plane for ring k
            z_tread = terrain_center[2] - (k + 1) * step_height
            lip_z   = (z_tread - lip_h / 2.0) + EPS   # top of lip aligns with tread; lip extends DOWN

            # INNER edge positions for ring k  (note the (k+1)!)
            y_top_in    = terrain_center[1] + terrain_size[1] / 2.0 - (k + 1) * cfg.step_width
            y_bottom_in = terrain_center[1] - terrain_size[1] / 2.0 + (k + 1) * cfg.step_width
            x_right_in  = terrain_center[0] + terrain_size[0] / 2.0 - (k + 1) * cfg.step_width
            x_left_in   = terrain_center[0] - terrain_size[0] / 2.0 + (k + 1) * cfg.step_width

            # Spans along the opening (use (k+1) so lips stop at corners cleanly)
            span_x = max(0.0, terrain_size[0] - 2.0 * (k + 1) * cfg.step_width)
            span_y = max(0.0, terrain_size[1] - 2.0 * (k + 1) * cfg.step_width)

            # +Y inner edge (protrude INTO trench: -Y direction)
            meshes_list.append(trimesh.creation.box(
                (span_x, lip_d, lip_h),
                trimesh.transformations.translation_matrix((
                    terrain_center[0],
                    y_top_in - lip_d / 2.0,   # center shifted inward by half the depth
                    lip_z
                ))
            ))

            # -Y inner edge (protrude +Y)
            meshes_list.append(trimesh.creation.box(
                (span_x, lip_d, lip_h),
                trimesh.transformations.translation_matrix((
                    terrain_center[0],
                    y_bottom_in + lip_d / 2.0,
                    lip_z
                ))
            ))

            # +X inner edge (protrude -X)
            meshes_list.append(trimesh.creation.box(
                (lip_d, span_y, lip_h),
                trimesh.transformations.translation_matrix((
                    x_right_in - lip_d / 2.0,
                    terrain_center[1],
                    lip_z
                ))
            ))

            # -X inner edge (protrude +X)
            meshes_list.append(trimesh.creation.box(
                (lip_d, span_y, lip_h),
                trimesh.transformations.translation_matrix((
                    x_left_in + lip_d / 2.0,
                    terrain_center[1],
                    lip_z
                ))
            ))



    # center platform box (unchanged)
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2.0)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
    return meshes_list, origin


@configclass
class MeshInvertedPyramidStairsWithLipTerrainCfg(MeshInvertedPyramidStairsTerrainCfg):
    function = inverted_pyramid_stairs_terrain_with_lip
    lip_height: float = 0.0
    lip_depth: float = 0.0
    lip_inset: float = 0.0
    add_bottom_lip: bool = True   # NEW

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.0,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "flat_mesh": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
            flat_patch_sampling=None,
        )
    },
)

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

ROUGH_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_26": MeshInvertedPyramidStairsWithLipTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            lip_height=0.02,     # 2 cm tall
            lip_depth=0.03,      # 3 cm total depth
            lip_inset=-0.02,     # 1 cm OUTSIDE edge (overhang), 1 cm inside
            add_bottom_lip=True, # lips on top AND bottom edges
        ),
        "pyramid_stairs_28": MeshInvertedPyramidStairsWithLipTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            lip_height=0.02,
            lip_depth=0.04,
            lip_inset=-0.03,
            add_bottom_lip=True,
        ),
        "pyramid_stairs_30_1": MeshInvertedPyramidStairsWithLipTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            lip_height=0.02,
            lip_depth=0.04,
            lip_inset=-0.03,
            add_bottom_lip=True,
        ),
        "pyramid_stairs_30_2": MeshInvertedPyramidStairsWithLipTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            lip_height=0.02,
            lip_depth=0.03,
            lip_inset=-0.02,
            add_bottom_lip=True,
        ),

        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.1, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.1, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=5,     
    num_cols=8,     
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat_corridors": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,  # 10% flat approach areas
            size=(1.5, 1.5),
        ),
        "regular_mesh_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.90,
            step_height_range=(0.11, 0.12),  # Fixed 11-12cm step height. Real steps are 10cm
            step_width=0.30,
            platform_width=1.8,
            border_width=0.6,
            holes=False,
        ),
    },
)