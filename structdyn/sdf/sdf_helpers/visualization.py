import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class SDFVisualizer:
    """
    Provides methods for visualizing a Single-Degree-of-Freedom (SDF) system.

    This class is not intended to be instantiated directly. Instead, it should be
    accessed through the `plot` property of an `SDF` instance (e.g., `sdf.plot.structure()`)
    which will be added in a subsequent step.
    """

    def __init__(self, sdf, tower_height=5.0, mass_size=(1.0, 0.5)):
        """
        Initializes the SDFVisualizer.

        Parameters
        ----------
        sdf : SDF
            The SDF object that this visualizer is attached to.
        tower_height : float, optional
            The height of the idealized tower, by default 5.0.
        mass_size : tuple, optional
            A tuple (width, height) for the mass block, by default (1.0, 0.5).
        """
        self.sdf = sdf
        self.tower_height = tower_height
        self.mass_width, self.mass_height = mass_size

    def _plot_displaced_sdf(
        self, ax, displacement, title="", max_disp_override=None, set_xlabel=False
    ):
        """Helper to plot the displaced shape of the SDF system on a given axis."""
        ax.clear()

        # Generate points for plotting a smooth column curve with double curvature
        num_points_curve = 20
        t_curve = np.linspace(0, 1, num_points_curve)
        # Cubic Hermite spline basis for zero-slope start/end points
        hermite_poly = -2 * t_curve**3 + 3 * t_curve**2

        # Y coordinates of the column
        y_curve = np.linspace(0, self.tower_height, num_points_curve)
        # X coordinates of the column, showing double curvature
        x_curve = displacement * hermite_poly

        # Plot tower column
        ax.plot(x_curve, y_curve, "b-")

        # Plot mass block on top of the column
        mass_x = displacement - self.mass_width / 2
        mass_y = self.tower_height  # Bottom of the mass is at the top of the column
        ax.add_patch(
            patches.Rectangle(
                (mass_x, mass_y), self.mass_width, self.mass_height, facecolor="gray"
            )
        )

        # Plot ground line
        ax.axhline(0, color="k", lw=2)

        # Formatting
        if max_disp_override is not None:
            max_disp = max_disp_override
        else:
            max_disp = abs(displacement)
            if max_disp == 0:
                max_disp = 1.0

        ax.set_xlim(-max_disp * 1.5 - self.mass_width, max_disp * 1.5 + self.mass_width)
        ax.set_ylim(-0.5, self.tower_height + self.mass_height * 2)
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        ax.set_yticks([])  # No y-ticks for this idealized diagram
        if set_xlabel:
            ax.set_xlabel("Displacement")

    def plot_system(self):
        """Plots the undeformed SDF system."""
        fig, ax = plt.subplots(figsize=(4, 6))
        self._plot_displaced_sdf(ax, 0, title="SDF System", set_xlabel=False)
        plt.show()

    def animate_response(
        self,
        response_df,
        scale_factor=1.0,
        ground_motion=None,
        speed_up=1.0,
        repeat=True,
        save_path=None,
    ):
        """
        Animates the dynamic response of the SDF system.

        Parameters
        ----------
        response_df : pandas.DataFrame
            The response DataFrame from a solver. Must contain 'time' and 'displacement' columns.
        scale_factor : float, optional
            Factor to scale displacements for visualization, by default 1.0.
        ground_motion : tuple, optional
            Tuple `(time, acceleration)` for the ground motion history. If provided,
            a plot of the ground motion is shown below the animation. By default None.
        speed_up : float, optional
            Factor to speed up the animation, by default 1.0.
        repeat : bool, optional
            Whether the animation should repeat when finished, by default True.
        save_path : str, optional
            File path to save the animation (e.g., 'animation.mp4'). If provided,
            the animation is saved instead of being shown. By default None.
        """
        if (
            "displacement" not in response_df.columns
            or "time" not in response_df.columns
        ):
            raise ValueError(
                "response_df must contain 'time' and 'displacement' columns."
            )

        displacements = response_df["displacement"].to_numpy()
        time_vector = response_df["time"].to_numpy()

        if ground_motion:
            fig, (ax_sys, ax_gm) = plt.subplots(
                2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]}
            )
            gm_time, gm_acc = ground_motion
            ax_gm.plot(gm_time, gm_acc, "k-")
            ax_gm.set_xlabel("Time (s)")
            ax_gm.set_ylabel("Ground Acc. (g)")
            ax_gm.grid(True)
            (time_marker,) = ax_gm.plot([], [], "r-", lw=2)
        else:
            fig, ax_sys = plt.subplots(figsize=(6, 6))

        max_abs_disp = np.max(np.abs(displacements)) * scale_factor

        def update(frame):
            frame_index = int(frame)
            current_time = time_vector[frame_index]
            current_disp_unscaled = displacements[frame_index]
            current_disp_scaled = current_disp_unscaled * scale_factor

            self._plot_displaced_sdf(
                ax_sys,
                current_disp_scaled,
                title=f"Deformed shape (SF = {scale_factor})",
                max_disp_override=max_abs_disp,
            )

            text_str = f"Time: {current_time:.2f} s\nDisp: {current_disp_unscaled:.4f}"
            ax_sys.text(
                0.05,
                0.5,
                text_str,
                transform=ax_sys.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            if ground_motion:
                ylim = ax_gm.get_ylim()
                time_marker.set_data([current_time, current_time], [ylim[0], ylim[1]])

        interval = (
            (time_vector[1] - time_vector[0]) * 1000 / speed_up
            if len(time_vector) > 1
            else 50
        )
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(time_vector)),
            interval=interval,
            repeat=repeat,
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            print(f"Saving animation to {save_path}... This may take a moment.")
            try:
                anim.save(save_path, writer="ffmpeg", dpi=150)
                print(f"Animation successfully saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print(
                    "Please ensure FFmpeg is installed and accessible in your system's PATH."
                )
            plt.close(fig)
        else:
            plt.show()

        return anim
