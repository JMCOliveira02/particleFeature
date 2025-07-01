import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import yaml

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def load_feature_map(yaml_file):
    """Load features from YAML file"""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('features', [])
    except:
        return []

def plot_feature(ax, feature, color_map):
    """Plot a single feature on the map"""
    x = feature['position']['x']
    y = feature['position']['y'] 
    theta = np.radians(feature['orientation']['theta'])
    feature_type = feature['type']
    
    color = color_map.get(feature_type, 'gray')
    
    if feature_type == 'table':
        # Draw table as rectangle (fixed orientation)
        width, height = 1.2, 0.8
        rect = patches.Rectangle((x-width/2, y-height/2), width, height, 
                               angle=np.degrees(theta), facecolor=color, 
                               edgecolor='black', alpha=0.7, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
    elif feature_type == 'chair':
        # Draw chair as circle with direction indicator (simpler and clearer)
        circle = patches.Circle((x, y), 0.3, facecolor=color, 
                              edgecolor='black', alpha=0.7, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, 'C', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Add small direction line instead of arrow
        line_len = 0.4
        dx = line_len * np.cos(theta)
        dy = line_len * np.sin(theta)
        ax.plot([x, x+dx], [y, y+dy], color='black', linewidth=3, alpha=0.9)
        
    # Skip doors, windows, and columns - only show tables and chairs

# Find CSV files in the current directory
csv_files = glob.glob('*.csv')

if not csv_files:
    print("❌ No CSV files found in the current directory!")
    print("Available files:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    exit(1)

print("📁 Available CSV files:")
for i, file in enumerate(csv_files):
    print(f"  {i+1}. {file}")

# Use the first CSV file or let user specify
if len(csv_files) == 1:
    csv_file = csv_files[0]
    print(f"🎯 Using: {csv_file}")
elif 'Nave_A_MM_semi.csv' in csv_files:
    csv_file = 'Nave_A_MM_semi.csv'
    print(f"🎯 Using default: {csv_file}")
else:
    csv_file = csv_files[0]
    print(f"🎯 Using first available: {csv_file}")

# Load CSV
try:
    df = pd.read_csv(csv_file)
    print(f"✅ Successfully loaded {csv_file} with {len(df)} rows")
except Exception as e:
    print(f"❌ Error loading {csv_file}: {e}")
    exit(1)

# Display column information
print(f"📊 CSV columns: {list(df.columns)}")
print(f"📏 Data shape: {df.shape}")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check for required columns
required_cols = ['timestamp_real', 'real_x', 'real_y', 'real_theta', 'est_x', 'est_y', 'est_theta']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Subtract initial time to normalize timestamps
df['timestamp_normalized'] = df['timestamp_real'] - df['timestamp_real'].iloc[0]

# Compute positional error
df['positional_error'] = np.sqrt((df['est_x'] - df['real_x'])**2 + (df['est_y'] - df['real_y'])**2)

# Compute angular error (more careful wrapping)
df['angular_error'] = np.abs(wrap_to_pi(df['est_theta'] - df['real_theta']))

# Detect when estimated pose is "stuck" (not changing)
df['est_position_change'] = np.sqrt(df['est_x'].diff()**2 + df['est_y'].diff()**2)
df['est_angle_change'] = np.abs(df['est_theta'].diff())
df['is_stuck'] = (df['est_position_change'] < 0.001) & (df['est_angle_change'] < 0.001)

# Fill NaN for first row
df['est_position_change'].fillna(0, inplace=True)
df['est_angle_change'].fillna(0, inplace=True)
df['is_stuck'].fillna(False, inplace=True)

# Count unique estimated poses
unique_poses = df[['est_x', 'est_y', 'est_theta']].drop_duplicates()
print(f"\n📍 Unique estimated poses: {len(unique_poses)} out of {len(df)} total samples")

# Basic stats
print("\n--- Positional Error ---")
print(f"Mean: {df['positional_error'].mean():.4f} m")
print(f"Max: {df['positional_error'].max():.4f} m")
print(f"Std: {df['positional_error'].std():.4f} m")

print("\n--- Angular Error ---")
print(f"Mean: {np.degrees(df['angular_error'].mean()):.2f}°")
print(f"Max: {np.degrees(df['angular_error'].max()):.2f}°")
print(f"Std: {np.degrees(df['angular_error'].std()):.2f}°")

print("\n--- Particle Filter Behavior ---")
stuck_percentage = (df['is_stuck'].sum() / len(df)) * 100
print(f"Estimate stuck (no change): {stuck_percentage:.1f}% of the time")

# Show the actual estimated poses
print(f"\nActual estimated poses found:")
for i, (_, row) in enumerate(unique_poses.iterrows()):
    if i < 10:  # Show first 10
        print(f"  Pose {i+1}: x={row['est_x']:.3f}, y={row['est_y']:.3f}, θ={np.degrees(row['est_theta']):.1f}°")
    elif i == 10:
        print(f"  ... and {len(unique_poses)-10} more")
        break

# Create three separate figures

# Figure 1: Positional Error
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp_normalized'], df['positional_error'], label='Positional Error (m)', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.title("Positional Error Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plot_filename_1 = csv_file.replace('.csv', '_positional_error.png')
plt.savefig(plot_filename_1, dpi=300, bbox_inches='tight')
print(f"📊 Positional error plot saved as: {plot_filename_1}")
plt.show()

# Figure 2: Angular Error
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp_normalized'], np.degrees(df['angular_error']), label='Angular Error (°)', color='orange', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Error (°)")
plt.title("Angular Error Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plot_filename_2 = csv_file.replace('.csv', '_angular_error.png')
plt.savefig(plot_filename_2, dpi=300, bbox_inches='tight')
print(f"📊 Angular error plot saved as: {plot_filename_2}")
plt.show()

# Figure 3: Trajectory Comparison
plt.figure(figsize=(10, 8))
# Plot real trajectory as continuous line
plt.plot(df['real_x'], df['real_y'], 'b-', label='Ground Truth Path', linewidth=3, alpha=0.8)

# Plot estimated trajectory - connect the points to show actual tracking
plt.plot(df['est_x'], df['est_y'], 'r--', label='Estimated Path', linewidth=2, alpha=0.7)

# Add start and end markers
plt.plot(df['real_x'].iloc[0], df['real_y'].iloc[0], 'go', markersize=12, 
         label='Start', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=2)
plt.plot(df['real_x'].iloc[-1], df['real_y'].iloc[-1], 'ro', markersize=12, 
         label='End', markerfacecolor='lightcoral', markeredgecolor='darkred', markeredgewidth=2)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory Comparison: Ground Truth vs Estimated")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plot_filename_3 = csv_file.replace('.csv', '_trajectory.png')
plt.savefig(plot_filename_3, dpi=300, bbox_inches='tight')
print(f"📊 Trajectory plot saved as: {plot_filename_3}")
plt.show()
