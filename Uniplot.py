"""
FINAL CORRECT VERSION - Real Meter Coordinates on Axes
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Load trajectory
DATA_DIR = Path("__data__/A3_PRAGMATIC_WORKING")
trajectory_file = DATA_DIR / "FINAL_trajectory.json"

with open(trajectory_file, 'r') as f:
    traj_data = json.load(f)

history = traj_data['history']
spawn_pos = traj_data.get('spawn_pos', [-0.8, 0, 0.1])
target_pos = traj_data.get('target_pos', [5, 0, 0.5])

x_coords = np.array([pos[0] for pos in history if len(pos) >= 2])
y_coords = np.array([pos[1] for pos in history if len(pos) >= 2])

print(f"\nTrajectory Summary:")
print(f"  Points: {len(history)}")
print(f"  X range: {x_coords.min():.2f} to {x_coords.max():.2f} meters")
print(f"  Y range: {y_coords.min():.2f} to {y_coords.max():.2f} meters")
print(f"  Start: ({x_coords[0]:.2f}, {y_coords[0]:.2f})")
print(f"  End: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f})")

# ================================================================
# VERSION 1: Black background with REAL METER COORDINATES
# ================================================================

fig, ax = plt.subplots(figsize=(7, 11), facecolor='black')
ax.set_facecolor('black')

# Work in REAL METER coordinates (not pixels!)
y_min = y_coords.min() - 1
y_max = y_coords.max() + 1

# Terrain sections in meters
# Smooth: -1 to 1.5m
smooth_rect = patches.Rectangle((-1, y_min), 2.5, y_max - y_min,
                                 facecolor='#808080', alpha=0.4)
ax.add_patch(smooth_rect)

# Checkerboard
checker_size_m = 0.25  # 25cm squares
for i in np.arange(-1, 1.5, checker_size_m):
    for j in np.arange(y_min, y_max, checker_size_m):
        if (int((i+1)/checker_size_m) + int((j-y_min)/checker_size_m)) % 2 == 0:
            checker = patches.Rectangle((i, j), checker_size_m, checker_size_m,
                                       facecolor='#404040', alpha=0.6, linewidth=0)
            ax.add_patch(checker)

# Rugged: 1.5 to 3.5m
rugged_rect = patches.Rectangle((1.5, y_min), 2.0, y_max - y_min,
                                 facecolor='#8B7355', alpha=0.5)
ax.add_patch(rugged_rect)

# Rock texture
np.random.seed(42)
y_span = y_max - y_min
n_rocks = int(200 * (y_span / 10))
rock_x = np.random.uniform(1.5, 3.5, n_rocks)
rock_y = np.random.uniform(y_min, y_max, n_rocks)
ax.scatter(rock_x, rock_y, s=6, c='#654321', alpha=0.6, marker='.')

# Uphill: 3.5 to 6m
uphill_rect = patches.Rectangle((3.5, y_min), 2.5, y_max - y_min,
                                 facecolor='#4169E1', alpha=0.4)
ax.add_patch(uphill_rect)

# Blue checkerboard
for i in np.arange(3.5, 6.0, checker_size_m):
    for j in np.arange(y_min, y_max, checker_size_m):
        if (int((i-3.5)/checker_size_m) + int((j-y_min)/checker_size_m)) % 2 == 0:
            checker = patches.Rectangle((i, j), checker_size_m, checker_size_m,
                                       facecolor='#1E3A8A', alpha=0.6, linewidth=0)
            ax.add_patch(checker)

# Plot path in REAL METER coordinates
ax.plot(x_coords, y_coords, 'b-', linewidth=3.5, label='Path', zorder=10)

# Markers
ax.plot(x_coords[0], y_coords[0], 'o', color='lime', markersize=16,
        label='Start', zorder=11, markeredgecolor='green', markeredgewidth=2)
ax.plot(x_coords[-1], y_coords[-1], 'o', color='red', markersize=16,
        label='End', zorder=11, markeredgecolor='darkred', markeredgewidth=2)

# Goal
if -1 <= target_pos[0] <= 6:
    ax.plot(target_pos[0], target_pos[1], '*', color='red', markersize=26,
            label='[0, 0, 0]', zorder=11, markeredgecolor='darkred', markeredgewidth=1.5)

# Set axis limits and labels
ax.set_xlim(-1.2, 6.2)
ax.set_ylim(y_max, y_min)  # Inverted for top-down view
ax.set_xlabel('X Position (m)', fontsize=13, color='white', fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=13, color='white', fontweight='bold')
ax.set_title('Robot Path in XY Plane', fontsize=15, color='white', fontweight='bold', pad=15)

# REAL METER TICK MARKS
x_ticks = np.arange(-1, 7, 1)  # -1, 0, 1, 2, 3, 4, 5, 6 meters
y_ticks = np.arange(int(y_min), int(y_max)+1, 2)  # Every 2 meters
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.tick_params(colors='white', labelsize=11)

# Grid for reference
ax.grid(True, alpha=0.2, linestyle=':', color='white', linewidth=0.5)

# Legend
legend = ax.legend(loc='upper right', fontsize=11, facecolor='black',
                  edgecolor='white', labelcolor='white')
legend.get_frame().set_alpha(0.85)

plt.tight_layout()
output_black = DATA_DIR / "robot_path_FINAL_BLACK.png"
plt.savefig(output_black, dpi=200, bbox_inches='tight', facecolor='black')
print(f"\nâœ… Black background version: {output_black}")
plt.close()

# ================================================================
# VERSION 2: White background - BEST FOR REPORT
# ================================================================

fig2, ax2 = plt.subplots(figsize=(10, 12))

# Terrain sections
smooth_patch = patches.Rectangle((-1, y_min), 2.5, y_max - y_min,
                                  facecolor='lightgray', alpha=0.5, label='Smooth Flat')
ax2.add_patch(smooth_patch)

rugged_patch = patches.Rectangle((1.5, y_min), 2.0, y_max - y_min,
                                  facecolor='wheat', alpha=0.6, label='Rugged Flat')
ax2.add_patch(rugged_patch)

uphill_patch = patches.Rectangle((3.5, y_min), 2.5, y_max - y_min,
                                  facecolor='lightblue', alpha=0.5, label='Smooth Uphill')
ax2.add_patch(uphill_patch)

# Add terrain labels
ax2.text(0.25, y_max - 0.5, 'Smooth\nFlat', ha='center', va='top',
         fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
ax2.text(2.5, y_max - 0.5, 'Rugged\nFlat', ha='center', va='top',
         fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
ax2.text(4.75, y_max - 0.5, 'Smooth\nUphill', ha='center', va='top',
         fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Path
ax2.plot(x_coords, y_coords, 'b-', linewidth=4, label='Robot Path', zorder=5)

# Markers
ax2.plot(x_coords[0], y_coords[0], 'o', color='green', markersize=17,
         label='Start', zorder=6, markeredgecolor='darkgreen', markeredgewidth=2.5)
ax2.plot(x_coords[-1], y_coords[-1], 'o', color='red', markersize=17,
         label='End', zorder=6, markeredgecolor='darkred', markeredgewidth=2.5)
ax2.plot(target_pos[0], target_pos[1], '*', color='red', markersize=30,
         label='Goal [0, 0, 0]', zorder=6, markeredgecolor='darkred', markeredgewidth=2)

# Distance annotation
dist = x_coords[-1] - x_coords[0]
ax2.annotate(f'Forward: {dist:.2f}m',
             xy=(x_coords[-1], y_coords[-1]), 
             xytext=(x_coords[-1] - 1.5, y_coords[-1] + 1.5),
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Formatting
ax2.set_xlim(-1.5, 6.5)
ax2.set_ylim(y_max, y_min)  # Inverted
ax2.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
ax2.set_title('Robot Path in XY Plane', fontsize=16, fontweight='bold', pad=20)

# Real meter ticks
ax2.set_xticks(np.arange(-1, 7, 1))
ax2.set_yticks(np.arange(int(y_min), int(y_max)+1, 2))
ax2.tick_params(axis='both', which='major', labelsize=12)

ax2.legend(loc='upper left', fontsize=12, framealpha=0.95, 
           edgecolor='black', shadow=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_aspect('equal', adjustable='box')

plt.tight_layout()
output_white = DATA_DIR / "robot_path_FINAL_WHITE.png"
plt.savefig(output_white, dpi=300, bbox_inches='tight', facecolor='white')
print(f"âœ… White background version: {output_white}")
plt.close()

print(f"\n{'='*70}")
print("PERFECT! Generated FINAL versions with CORRECT coordinates:")
print(f"{'='*70}")
print(f"1. robot_path_FINAL_BLACK.png")
print(f"   â†’ Black background (uni style)")
print(f"   â†’ REAL METER coordinates on axes")
print(f"   â†’ X: -1, 0, 1, 2, 3, 4, 5, 6 (meters)")
print(f"   â†’ Y: Shows actual meter positions")
print(f"\n2. robot_path_FINAL_WHITE.png â­ RECOMMENDED FOR REPORT")
print(f"   â†’ Clean white background")
print(f"   â†’ Terrain labels visible")
print(f"   â†’ Distance annotation")
print(f"   â†’ High resolution (300 DPI)")
print(f"   â†’ Perfect for printing")
print(f"\n{'='*70}")
print("Path Performance:")
print(f"{'='*70}")
print(f"Forward distance: {x_coords[-1] - x_coords[0]:.2f} meters")
print(f"Final X position: {x_coords[-1]:.2f}m")
print(f"Terrain reached: ", end='')
if x_coords[-1] > 3.5:
    print("âœ“ UPHILL SECTION (3.5m+)")
elif x_coords[-1] > 1.5:
    print("âœ“ RUGGED SECTION (1.5-3.5m)")
else:
    print("SMOOTH SECTION (0-1.5m)")
print(f"\nPercentage of course: {(x_coords[-1] / 5.8) * 100:.1f}%")
print(f"{'='*70}")