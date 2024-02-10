# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:30:14 2024

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:01:25 2024

@author: logan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

#%%

# Load the spacecraft data
df = pd.read_csv('imap_traj_data.csv')

Re = 6378

# Convert 'Time (UTCG)' to datetime format
df['Time (UTCG)'] = pd.to_datetime(df['Time (UTCG)'])

# Set the desired aspect ratio for the 3D plot
aspect_ratio = [0.75,0.8,1]

# Plot the 3D trajectory
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')

ax.plot(df['x (km)']/Re, df['y (km)']/Re, df['z (km)']/Re, label='Trajectory')

# Set equal scales in all three planes within the 3D plot
ax.set_box_aspect(aspect_ratio)

# Labeling
ax.set_xlabel('x ($R_e$)')
ax.set_ylabel('y ($R_e$)')
ax.set_zlabel('z ($R_e$)')
#ax.set_title('3D Trajectory of Spacecraft')
##ax.legend()

plt.quiver(df['x (km)'][100]/Re,df['y (km)'][100]/Re,df['z (km)'][100]/Re,df['x (km/sec)'][100],
           df['y (km/sec)'][100],df['z (km/sec)'][100],zorder=10,color='black',pivot='middle',length=0.4)

# Adjust subplot parameters to add more space around the plot
#plt.subplots_adjust(left=-1) # Does nothing

# Adjust the viewing angle or elevation
#ax.view_init(elev=30, azim=-30)

ax.set_zlim(-15, 15)

# Set ticks for x, y, and z axes
ax.set_xticks(range(210, 250, 10))
ax.set_yticks(range(-40, 40, 20))
ax.set_zticks(range(-15, 16, 5))

# Show the plot
plt.show()

#%% Swapping axes for better visuals

#plt.figure(figsize=(3,4))


# Plot with equal aspect ratio, adjusted colors and style
plt.plot(df['y (km)']/Re, df['x (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
plt.axis('equal')  # Set equal aspect ratio for x and y axes

# Labeling
plt.xlabel('y ($R_e$)')
plt.ylabel('x ($R_e$)')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)
plt.quiver(df['y (km)'][100]/Re,df['x (km)'][100]/Re,df['y (km/sec)'][100]/Re,df['x (km/sec)'][100]/Re,
           zorder=10)
# Show the plot
plt.show()

# Plot with equal aspect ratio, adjusted colors and style
plt.plot(df['y (km)']/Re, df['z (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
plt.axis('equal')  # Set equal aspect ratio for x and y axes

# Labeling
plt.xlabel('y ($R_e$)')
plt.ylabel('z ($R_e$)')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)
plt.quiver(df['y (km)'][100]/Re,df['z (km)'][100]/Re,df['y (km/sec)'][100]/Re,df['z (km/sec)'][100]/Re,
           zorder=10)
# Show the plot
plt.show()

# Plot with equal aspect ratio, adjusted colors and style
plt.plot(df['x (km)']/Re, df['z (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
plt.axis('equal')  # Set equal aspect ratio for x and y axes

# Labeling
plt.xlabel('x ($R_e$)')
plt.ylabel('z ($R_e$)')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)
plt.quiver(df['x (km)'][100]/Re,df['z (km)'][100]/Re,df['x (km/sec)'][100]/Re,df['z (km/sec)'][100]/Re,
           zorder=10)
# Show the plot
plt.show()



#%%

arsize = 3
# Function to add arrowheads based on velocity at evenly spaced points
def add_velocity_arrowheads(ax, x, y, u, v, num_arrows=10, arrow_size=10):
    idx = np.linspace(0, len(x) - 1, num_arrows, dtype=int)
    ax.quiver(x[idx], y[idx], u[idx], v[idx], scale_units='xy', angles='xy', scale=arrow_size, color='r', 
              label='Velocity', pivot='middle', headaxislength=arsize, headlength=arsize, headwidth=arsize, 
              minshaft=0)

# Plot trajectory for XY plane
fig, ax_xy = plt.subplots(figsize=(8, 8))
ax_xy.plot(df['y (km)']/Re, df['x (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
add_velocity_arrowheads(ax_xy, df['y (km)'].values/Re, df['x (km)'].values/Re, df['y (km/sec)'].values, 
                        df['x (km/sec)'].values)
ax_xy.axis('equal')  # Set equal aspect ratio for x and y axes
ax_xy.set_xlabel('y ($R_e$)')
ax_xy.set_ylabel('x ($R_e$)')
ax_xy.set_title('XY Plane')
ax_xy.legend()
plt.show()

# Plot trajectory for XZ plane
fig, ax_xz = plt.subplots(figsize=(8, 8))
ax_xz.plot(df['z (km)']/Re, df['x (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
add_velocity_arrowheads(ax_xz, df['z (km)'].values/Re, df['x (km)'].values/Re, df['z (km/sec)'].values, 
                        df['x (km/sec)'].values)
ax_xz.axis('equal')  # Set equal aspect ratio for x and z axes
ax_xz.set_xlabel('z ($R_e$)')
ax_xz.set_ylabel('x ($R_e$)')
ax_xz.set_title('XZ Plane')
ax_xz.legend()
plt.show()

# Plot trajectory for YZ plane
fig, ax_yz = plt.subplots(figsize=(8, 8))
ax_yz.plot(df['z (km)']/Re, df['y (km)']/Re, linestyle='-', marker='o', markersize=1, label='Trajectory')
add_velocity_arrowheads(ax_yz, df['z (km)'].values/Re, df['y (km)'].values/Re, df['z (km/sec)'].values, 
                        df['y (km/sec)'].values)
ax_yz.axis('equal')  # Set equal aspect ratio for y and z axes
ax_yz.set_xlabel('z ($R_e$)')
ax_yz.set_ylabel('y ($R_e$)')
ax_yz.set_title('YZ Plane')
ax_yz.legend()
plt.show()





