# ================================================================
#  场景资产注册（Manager-Based 架构）
#  可直接在 env_cfg.py 中通过 SceneAssetsCfg 引用
# ================================================================
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg  # 新增导入
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
# 执行器
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class SceneAssetsCfg(InteractiveSceneCfg):
    # --- 背景 ---
    ground = AssetBaseCfg(
        prim_path="/World/ground", # 默认地面路径
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)), # 生成指定大小的地面
    )

    # --- 灯光 ---
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", # 默认灯光路径
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)), # 设置灯光强度与颜色
    )


    # --- 1. 机器人 (删除 sensors 参数) ---
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/first_rl/first_rl/tasks/manager_based/first_rl/IsaacSim_Assets/" \
                     "Collected_so101_new_calib_physics/so101_new_calib_physics.usd",
            activate_contact_sensors=True,
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.4, 0.0, 0.446),
            rot=(0.707, 0.0, 0.0, 0.707),
            joint_pos={
                "shoulder_pan": 0.0, "shoulder_lift": -0.5, "elbow_flex": 0.5, 
                "wrist_flex": 0.0, "wrist_roll": 0.5, "gripper": 0.08,
            },
        ),
        actuators={
            "so101_arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                effort_limit_sim=400.0, velocity_limit_sim=2.175, stiffness=800.0, damping=50.0,
            ),
            "so101_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=100.0, velocity_limit_sim=2.0, stiffness=400.0, damping=10.0,
            ),
        },
    )

    # --- 桌子 ---
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.5), # 长方体桌子的尺寸
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)), # 灰色材质
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),     # 使用默认刚体属性
            collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞属性
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)), # 桌子初始位置
    )

    # --- 红色正方体 ---
    # 定义可交互的刚体对象：设置尺寸(5cm)、红色材质、质量(0.1kg)及碰撞
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), # 5cm正方体
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # 红色
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 默认刚体属性
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1), # 设置质量为0.1kg
            collision_props=sim_utils.CollisionPropertiesCfg() # 开启碰撞
        ),
        # 正方体在桌子上的初始位置
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.526)), 
    )