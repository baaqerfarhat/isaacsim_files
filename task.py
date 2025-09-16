from dataclasses import MISSING
from typing import Sequence

import torch
import sys
import threading
import queue

from srb import assets
from srb._typing import StepReturn
from srb.core.action import ThrustAction
from srb.core.asset import AssetBaseCfg
from srb.core.env import OrbitalEnv, OrbitalEnvCfg, OrbitalEventCfg, OrbitalSceneCfg
from srb.utils.cfg import configclass
from srb.utils.math import matrix_from_quat, rotmat_to_rot6d
from srb.core.sim import UsdFileCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.sim import PreviewSurfaceCfg, SphereCfg

##############
### Config ###
##############


@configclass
class SceneCfg(OrbitalSceneCfg):
    env_spacing: float = 25.0


@configclass
class EventCfg(OrbitalEventCfg):
    # No randomization or movement
    pass


@configclass
class TaskCfg(OrbitalEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()
    ## Robot - Use Custom Cube spacecraft
    robot = assets.CustomCube()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 30.0
    is_finite_horizon: bool = False
    # Prevent automatic resets that could snap pose back
    truncate_episodes: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Single static Apophis asteroid at origin (no physics)
        self.scene.apophis = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/apophis",
            spawn=UsdFileCfg(
                usd_path="/root/ws/assets/custom/apophis.usdc",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )


############
### Task ###
############


class Task(OrbitalEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## No dynamic objects; robot-only scene for manual testing

        ## Place spacecraft at origin for easy viewing in the GUI
        # Start 20 m away from Apophis (origin). Place along +X for visibility.
        start_pos = torch.tensor([20.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        # Face the asteroid at the origin: yaw 180 deg so +X points toward -X world
        start_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        pos_w = self.scene.env_origins + start_pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat_w = start_quat.unsqueeze(0).repeat(self.num_envs, 1)
        # Write initial pose and zero velocity to simulation
        self._static_pose = torch.cat([pos_w, quat_w], dim=-1)
        self._static_velocity = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._robot.write_root_pose_to_sim(self._static_pose)
        self._robot.write_root_velocity_to_sim(self._static_velocity)

        ## Find thrust term for simple internal demo control
        self._thrust_term = None
        for term in self.action_manager._terms.values():  # type: ignore
            if isinstance(term, ThrustAction):
                self._thrust_term = term
                break

        # User interactive control (default: no movement until commanded)
        # VenusExpress thruster order: [ +X, -X, +Y, -Y ]
        self._dir_to_thruster_idx = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}
        self._active_thruster_idx = None
        self._active_thruster_mag: float = 0.0
        self._user_steps_remaining: int = 0
        print("Interactive control active. You'll be prompted for direction, magnitude, and steps.", flush=True)
        print("Directions: +x, -x, +y, -y. Type 'stop' to cancel, 'q' to stay idle.", flush=True)
        # Start a background stdin reader so GUI is not blocked
        self._input_queue: "queue.Queue[str]" = queue.Queue()
        self._input_thread = threading.Thread(target=self._stdin_reader, daemon=True)
        self._input_thread.start()
        
        # Track when we transition from active thrust to idle
        self._was_active_last_step: bool = False

        # Orbit tracker (toggle via CLI: 'mpc on' / 'mpc off')
        self._mpc_enabled: bool = False
        self._mpc_radius: float = 40.0
        self._mpc_omega: float = 0.6  # rad/s (slightly slower, easier to track)
        # Orbit centered at Apophis (origin)
        self._mpc_center_xy = torch.tensor([0.0, 0.0], device=self.device)
        self._mpc_kp: float = 0.4
        self._mpc_kd: float = 0.8
        # Map desired acceleration [m/s^2] to action [0..1]
        self._mpc_acc_to_action: float = 0.15
        self._mpc_time: float = 0.0
        # Receding-horizon MPC (finite-horizon LQR tracking) parameters
        self._mpc_horizon_steps: int = 60
        self._mpc_q_pos: float = 5.0
        self._mpc_q_vel: float = 2.0
        self._mpc_r_u: float = 0.05
        # Tangential speed ramp to avoid initial saturation
        self._mpc_speed_ramp_time_s: float = 3.0
        # Orbit controller gains and state
        self._orb_k_t: float = 0.9   # tangential speed tracking
        self._orb_k_r: float = 0.8   # radial position correction
        self._orb_k_dr: float = 1.1  # radial velocity damping
        self._orb_k_i: float = 0.05  # radial integral gain
        self._orb_e_r_int: float = 0.0
        self._orb_e_r_int_limit: float = 2.0
        # Action smoothing to avoid rapid flips when saturating
        self._action_smoothing: float = 0.2
        self._last_actions: torch.Tensor | None = None
        # Orbit completion tracking
        self._orbit_start_pos_xy = None  # type: ignore[var-annotated]
        self._orbit_last_angle: float = 0.0
        self._orbit_angle_accum: float = 0.0
        self._orbit_close_ready: bool = False
        # Mode selection: manual by default until user picks
        self._mode_selected: bool = False

        # Trajectory trace (breadcrumb spheres)
        self._traj_enabled: bool = True
        self._traj_interval_steps: int = 5
        self._traj_step_counter: int = 0
        self._traj_next_id: int = 0

        # Keep IMU enabled so ROS can publish /sc4/Imu


    def _reset_idx(self, env_ids: Sequence[int]):
        # No-op reset: keep current pose; just zero velocities to stabilize
        self._robot.write_root_velocity_to_sim(self._static_velocity)

    def extract_step_return(self) -> StepReturn:
        # Interactive CLI thrust control: poll non-blocking command queue
        if self._thrust_term is not None:
            actions = torch.zeros((self.num_envs, self._thrust_term.action_dim), device=self.device)
            if self._mpc_enabled:
                actions = self._compute_orbit_actions()
                active_now = True
                self._mpc_time += float(self.step_dt)
            else:
                if self._user_steps_remaining <= 0:
                    self._poll_user_command_nonblocking()

                active_now = self._active_thruster_idx is not None and self._user_steps_remaining > 0

                if active_now:
                    if self._thrust_term.action_dim >= 8:
                        idx = int(self._active_thruster_idx)  # 0:+X, 1:-X, 2:+Y, 3:-Y
                        mag = float(self._active_thruster_mag)
                        if idx == 0:
                            actions[:, 0] = mag
                            actions[:, 1] = mag
                        elif idx == 1:
                            actions[:, 2] = mag
                            actions[:, 3] = mag
                        elif idx == 2:
                            actions[:, 4] = mag
                            actions[:, 5] = mag
                        elif idx == 3:
                            actions[:, 6] = mag
                            actions[:, 7] = mag
                    else:
                        actions[:, self._active_thruster_idx] = self._active_thruster_mag
                    self._user_steps_remaining -= 1
                else:
                    # Keep absolutely static when idle (no drift). This does not change pose.
                    self._robot.write_root_velocity_to_sim(self._static_velocity)

            self._was_active_last_step = active_now

            self._thrust_term.process_actions(actions)
            self._thrust_term.apply_actions()

        # Trajectory breadcrumbs
        if self._traj_enabled:
            self._traj_step_counter += 1
            if self._traj_step_counter % self._traj_interval_steps == 0:
                self._spawn_traj_dot()

        # Minimal step return without rewards or targets
        num_envs = self.num_envs
        termination = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        truncation = (
            self.episode_length_buf >= self.max_episode_length
            if self.cfg.truncate_episodes
            else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        )
        

        observation = {
            "state": {
                "tf_rot6d_robot": rotmat_to_rot6d(matrix_from_quat(self._robot.data.root_quat_w)),
                "vel_lin_robot": self._robot.data.root_lin_vel_b,
                "vel_ang_robot": self._robot.data.root_ang_vel_b,
            },
        }
        reward = {"zero": torch.zeros(num_envs, device=self.device)}
        return StepReturn(observation, reward, termination, truncation)


## Removed reward/target computation: task is static spawn only

    def _prompt_user_thrust(self) -> None:
        """Blocking CLI prompt to get user thrust command.
        Sets internal state for which thruster to fire, magnitude, and duration (in steps).
        """
        try:
            print("[UserControl] Enter direction (+x, -x, +y, -y, stop). 'q' to keep idle:", flush=True)
            direction = input(">> direction: ").strip().lower()
            if direction in ("q", "quit", "exit", ""):
                self._active_thruster_idx = None
                self._active_thruster_mag = 0.0
                self._user_steps_remaining = 0
                print("[UserControl] Idle. No thrust.")
                return
            if direction == "stop":
                self._active_thruster_idx = None
                self._active_thruster_mag = 0.0
                self._user_steps_remaining = 0
                print("[UserControl] Stopped. No thrust.")
                return
            if direction not in self._dir_to_thruster_idx:
                print(f"[UserControl] Invalid direction '{direction}'. No thrust.")
                self._active_thruster_idx = None
                self._active_thruster_mag = 0.0
                self._user_steps_remaining = 0
                return
            mag_str = input(">> magnitude [0..1]: ").strip()
            try:
                mag = float(mag_str)
            except Exception:
                mag = 0.0
            mag = max(0.0, min(1.0, mag))
            steps_str = input(">> steps (positive integer): ").strip()
            try:
                steps = int(steps_str)
            except Exception:
                steps = 0
            steps = max(0, steps)
            if steps <= 0 or mag <= 0.0:
                self._active_thruster_idx = None
                self._active_thruster_mag = 0.0
                self._user_steps_remaining = 0
                print("[UserControl] Zero-duration or zero-magnitude. No thrust.")
                return
            self._active_thruster_idx = self._dir_to_thruster_idx[direction]
            self._active_thruster_mag = mag
            self._user_steps_remaining = steps
            print(f"[UserControl] Applying {mag:.3f} thrust on '{direction}' for {steps} steps.")
        except Exception as e:
            print(f"[UserControl] Input error: {e}. No thrust.")
            self._active_thruster_idx = None
            self._active_thruster_mag = 0.0
            self._user_steps_remaining = 0

    def _stdin_reader(self) -> None:
        """Background thread: interactively prompts for direction, magnitude, and steps and enqueues a command string."""
        while True:
            try:
                if not self._mode_selected:
                    print("Select control mode [mpc/manual] (default manual):", flush=True)
                    mode = input(">> mode: ").strip().lower()
                    if mode == "mpc":
                        self._input_queue.put("mpc on")
                        self._mode_selected = True
                        continue
                    # default to manual
                    self._mode_selected = True
                print("Enter direction (+x, -x, +y, -y). 'stop' to cancel, 'q' to stay idle:", flush=True)
                direction = input(">> direction: ").strip().lower()
                if direction in ("mpc on", "mpc off", "mpc"):
                    self._input_queue.put(direction)
                    continue
                if direction in ("cam on", "cam off", "camera on", "camera off", "view on", "view off"):
                    self._input_queue.put(direction)
                    continue
                if direction in ("manual", "man"):
                    self._input_queue.put("mpc off")
                    continue
                if direction in ("q", "quit", "exit", ""):
                    self._input_queue.put("q")
                    continue
                if direction == "stop":
                    self._input_queue.put("stop")
                    continue
                if direction not in self._dir_to_thruster_idx:
                    print("Invalid direction. Use +x, -x, +y, -y.", flush=True)
                    continue
                mag_str = input(">> magnitude [0..1] (default 0.2): ").strip()
                if mag_str == "":
                    mag_val = 0.2
                else:
                    try:
                        mag_val = float(mag_str)
                    except Exception:
                        mag_val = 0.0
                steps_str = input(">> steps (positive integer) (default 50): ").strip()
                if steps_str == "":
                    steps_val = 50
                else:
                    try:
                        steps_val = int(steps_str)
                    except Exception:
                        steps_val = 0
                self._input_queue.put(f"{direction} {mag_val} {steps_val}")
            except Exception:
                # Keep the reader alive even if stdin glitches
                continue

    def _poll_user_command_nonblocking(self) -> None:
        """Poll the input queue and update active thrust command if a new line is available."""
        try:
            while True:
                cmd = self._input_queue.get_nowait()
                if not cmd:
                    continue
                cmd = cmd.strip().lower()
                if cmd in ("mpc on", "mpc off", "mpc"):
                    self._mpc_enabled = cmd == "mpc on" or cmd == "mpc"
                    if self._mpc_enabled:
                        # Restart phase and orbit tracking state
                        self._mpc_time = 0.0
                        # Record start position and angle (env 0)
                        p0 = self._robot.data.root_pos_w[0, :2]
                        # Keep orbit centered at origin (Apophis)
                        center = self._mpc_center_xy
                        r_vec0 = p0 - center
                        self._orbit_start_pos_xy = p0.detach()
                        self._orbit_last_angle = float(torch.atan2(r_vec0[1], r_vec0[0]))
                        self._orbit_angle_accum = 0.0
                        self._orb_e_r_int = 0.0
                        self._orbit_close_ready = False
                    # clear any pending manual command
                    self._active_thruster_idx = None
                    self._active_thruster_mag = 0.0
                    self._user_steps_remaining = 0
                    continue
                if cmd in ("cam on", "camera on", "view on"):
                    cam_path = f"/World/envs/env_0/robot/camera_onboard"
                    self._set_viewport_camera(cam_path)
                    continue
                if cmd in ("cam off", "camera off", "view off"):
                    self._set_viewport_camera(None)
                    continue
                if cmd in ("manual", "man"):
                    self._mpc_enabled = False
                    # clear pending states
                    self._active_thruster_idx = None
                    self._active_thruster_mag = 0.0
                    self._user_steps_remaining = 0
                    continue
                if cmd in ("q", "quit", "exit"):
                    self._active_thruster_idx = None
                    self._active_thruster_mag = 0.0
                    self._user_steps_remaining = 0
                    continue
                if cmd == "stop":
                    self._active_thruster_idx = None
                    self._active_thruster_mag = 0.0
                    self._user_steps_remaining = 0
                    continue
                parts = cmd.split()
                if len(parts) not in (1, 3):
                    continue
                direction = parts[0]
                if direction not in self._dir_to_thruster_idx:
                    continue
                if len(parts) == 1:
                    magnitude = 0.2
                    steps = 50
                else:
                    try:
                        magnitude = float(parts[1])
                    except Exception:
                        magnitude = 0.0
                    try:
                        steps = int(parts[2])
                    except Exception:
                        steps = 0
                magnitude = max(0.0, min(1.0, magnitude))
                steps = max(0, steps)
                if steps <= 0 or magnitude <= 0.0:
                    self._active_thruster_idx = None
                    self._active_thruster_mag = 0.0
                    self._user_steps_remaining = 0
                    continue
                self._active_thruster_idx = self._dir_to_thruster_idx[direction]
                self._active_thruster_mag = magnitude
                self._user_steps_remaining = steps
        except queue.Empty:
            # No command available; nothing to do
            return

    def _compute_orbit_actions(self) -> torch.Tensor:
        """Geometric orbit controller: centripetal + tangential speed tracking with radial correction."""
        # State
        pos_w = self._robot.data.root_pos_w[:, :2]
        vel_b = self._robot.data.root_lin_vel_b
        R_wb = matrix_from_quat(self._robot.data.root_quat_w)
        vel_b3 = torch.zeros((self.num_envs, 3), device=self.device)
        vel_b3[:, :] = vel_b
        vel_w3 = torch.bmm(R_wb, vel_b3.unsqueeze(-1)).squeeze(-1)
        vel_w = vel_w3[:, :2]

        # Orbit geometry
        center = self._mpc_center_xy
        r_vec = pos_w - center.unsqueeze(0)
        r = torch.clamp(torch.linalg.norm(r_vec, dim=1), min=1e-5)
        u_r = r_vec / r.unsqueeze(1)
        # Tangent (CCW): rotate u_r by +90 deg
        u_t = torch.stack([-u_r[:, 1], u_r[:, 0]], dim=1)

        # Speeds and ramp
        w = self._mpc_omega
        Rref = self._mpc_radius
        v_des_t = w * Rref
        Tr = max(self._mpc_speed_ramp_time_s, 1e-6)
        t_now = float(self._mpc_time)
        ramp_alpha = min(1.0, t_now / Tr)
        ramp_alpha_dot = (1.0 / Tr) if t_now < Tr else 0.0
        v_des_t_eff = v_des_t * ramp_alpha
        v_t = torch.sum(vel_w * u_t, dim=1)
        v_r = torch.sum(vel_w * u_r, dim=1)
        e_r = r - Rref

        # Desired acceleration in world (FF + PD)
        # Feed-forward centripetal using desired tangential speed
        a_c_ff = (v_des_t_eff * v_des_t_eff) / r
        a_world = (-a_c_ff).unsqueeze(1) * u_r  # centripetal toward center (FF)
        # Tangential feed-forward during ramp (accelerate along tangent)
        a_t_ff = (v_des_t * ramp_alpha_dot)
        a_world = a_world + a_t_ff * u_t
        # Tangential PD on speed tracking
        a_world = a_world + self._orb_k_t * (v_des_t_eff - v_t).unsqueeze(1) * u_t
        
        # Radial PID (with simple anti-windup)
        # Update integral only if radial error is small enough and we are not likely saturating
        e_r_mean = float(torch.mean(e_r))
        self._orb_e_r_int = float(max(-self._orb_e_r_int_limit, min(self._orb_e_r_int + e_r_mean * float(self.step_dt), self._orb_e_r_int_limit)))
        a_world = a_world + (-self._orb_k_r * e_r).unsqueeze(1) * u_r + (-self._orb_k_dr * v_r).unsqueeze(1) * u_r
        a_world = a_world + (-self._orb_k_i * self._orb_e_r_int) * u_r

        # Limit commanded acceleration magnitude to avoid thruster saturation
        a_norm = torch.clamp(torch.linalg.norm(a_world, dim=1), min=1e-6)
        a_max_cmd = 1.0 / max(self._mpc_acc_to_action, 1e-6)
        scale_limiter = torch.clamp(a_max_cmd / a_norm, max=1.0)
        a_world = a_world * scale_limiter.unsqueeze(1)

        # Orbit completion detection (env 0): accumulate angle traveled
        if self._orbit_start_pos_xy is not None:
            r0 = r_vec[0]
            angle = float(torch.atan2(r0[1], r0[0]))
            dtheta = float(torch.atan2(
                torch.sin(torch.tensor(angle - self._orbit_last_angle)),
                torch.cos(torch.tensor(angle - self._orbit_last_angle)),
            ))
            self._orbit_angle_accum += dtheta
            self._orbit_last_angle = angle
            # Close when passed ~2π and near start pos
            if abs(self._orbit_angle_accum) >= 2.0 * 3.14159 - 0.2:
                p_now = pos_w[0]
                dist_to_start = float(torch.linalg.norm(p_now - self._orbit_start_pos_xy.to(self.device)))
                if dist_to_start < 0.3:
                    self._orbit_close_ready = True

        # Transform to body
        R_bw = R_wb.transpose(1, 2)
        a_w3 = torch.zeros((self.num_envs, 3), device=self.device)
        a_w3[:, 0:2] = a_world
        a_b3 = torch.bmm(R_bw, a_w3.unsqueeze(-1)).squeeze(-1)
        ax = a_b3[:, 0]
        ay = a_b3[:, 1]

        # Map to thrusters
        scale = self._mpc_acc_to_action
        act_px = torch.clamp(scale * torch.relu(ax), 0.0, 1.0)
        act_nx = torch.clamp(scale * torch.relu(-ax), 0.0, 1.0)
        act_py = torch.clamp(scale * torch.relu(ay), 0.0, 1.0)
        act_ny = torch.clamp(scale * torch.relu(-ay), 0.0, 1.0)
        actions4 = torch.stack((act_px, act_nx, act_py, act_ny), dim=1)
        # Build action vector, smooth, and stop if orbit complete
        if self._thrust_term.action_dim >= 8:
            actions = torch.zeros((self.num_envs, self._thrust_term.action_dim), device=self.device)
            actions[:, 0] = actions4[:, 0]
            actions[:, 1] = actions4[:, 0]
            actions[:, 2] = actions4[:, 1]
            actions[:, 3] = actions4[:, 1]
            actions[:, 4] = actions4[:, 2]
            actions[:, 5] = actions4[:, 2]
            actions[:, 6] = actions4[:, 3]
            actions[:, 7] = actions4[:, 3]
        else:
            actions = actions4

        # Smooth actions to improve robustness
        if self._last_actions is None or self._last_actions.shape != actions.shape:
            self._last_actions = torch.zeros_like(actions)
        s = float(self._action_smoothing)
        actions = (1.0 - s) * actions + s * self._last_actions
        self._last_actions = actions

        # Stop cleanly once orbit closes (env 0)
        if self._orbit_close_ready:
            actions = torch.zeros_like(actions)
            self._mpc_enabled = False
        return actions

    def _set_viewport_camera(self, camera_prim_path: str | None) -> None:
        """Switch active viewport to given camera prim path (env 0), or back to free camera when None."""
        try:
            try:
                import omni.kit.viewport.utility as vp_utils  # type: ignore
                vp = vp_utils.get_active_viewport_window()
                if vp is not None:
                    if camera_prim_path:
                        vp.set_active_camera_path(camera_prim_path)
                    else:
                        # Reset to free camera
                        vp.set_active_camera_path("")
                    return
            except Exception:
                pass
            # Legacy fallback
            try:
                import omni.kit.viewport_legacy as vp_legacy  # type: ignore
                vw = vp_legacy.get_default_viewport_window()
                if vw is not None:
                    if camera_prim_path:
                        vw.set_active_camera(camera_prim_path)
                    else:
                        vw.set_active_camera(None)
            except Exception:
                pass
        except Exception:
            # Non-fatal if viewport control fails
            return

    def _compute_lqr_gain_1d(self, dt: float, horizon: int, q_pos: float, q_vel: float, r_u: float) -> torch.Tensor:
        """Finite-horizon LQR gain for 1D double-integrator with state [p, v], input [a].
        Returns K0 of shape (1,2) such that u = -K0 @ [e_p, e_v].
        """
        # System
        A = torch.tensor([[1.0, dt], [0.0, 1.0]], device=self.device)
        B = torch.tensor([[0.5 * dt * dt], [dt]], device=self.device)
        Q = torch.diag(torch.tensor([q_pos, q_vel], device=self.device))
        R = torch.tensor([[r_u]], device=self.device)
        S = Q.clone()  # terminal cost
        # Backward Riccati to get K0; store only next S
        for _ in range(horizon):
            BT_S = B.T @ S
            inv_term = torch.linalg.inv(R + BT_S @ B)
            K = inv_term @ (BT_S @ A)
            # Update S
            S = Q + A.T @ S @ (A - B @ K)
        # K computed at final iteration corresponds to K0
        return K

    def _spawn_traj_dot(self) -> None:
        try:
            pos_w = self._robot.data.root_pos_w[:, :3]
            quat_w = self._robot.data.root_quat_w
            # Use env 0 for visualization
            p = pos_w[0].detach().cpu()
            q = quat_w[0].detach().cpu()
            prim_path = f"/Visuals/traj/dot_{self._traj_next_id}"
            cfg = VisualizationMarkersCfg(
                prim_path=prim_path,
                markers={
                    "dot": SphereCfg(
                        radius=0.05,
                        visual_material=PreviewSurfaceCfg(emissive_color=(1.0, 0.2, 0.2)),
                    )
                },
            )
            marker = VisualizationMarkers(cfg)
            marker.visualize(p.unsqueeze(0), q.unsqueeze(0))
            self._traj_next_id += 1
        except Exception:
            pass
