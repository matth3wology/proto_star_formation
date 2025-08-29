import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Simulation Parameters
NUM_PARTICLES = 500
SPACE_SIZE = 120
G = 1.0
TIME_STEP = 0.03
MERGE_DISTANCE = 2.0
PROB_GROWTH = 0.08
FUSION_THRESHOLD = 6.0
ROTATION_SPEED = 0.015
PRESSURE_COEFF = 0.04
FEEDBACK_STRENGTH = 0.06
BINARY_SEPARATION = 25.0  # minimum separation for binary formation
TIDAL_FORCE_COEFF = 0.02  # strength of tidal interactions

# Initialize Particles
# Each particle: [x, y, z, vx, vy, vz, mass, type, binary_id]
# type: 0=H, 1=He, 2=Dust
# binary_id: 0=unbound, 1=primary, 2=secondary
particles = np.zeros((NUM_PARTICLES, 9))

# Create two initial clumps for binary formation
center1 = np.array([SPACE_SIZE/3, SPACE_SIZE/2, SPACE_SIZE/2])
center2 = np.array([2*SPACE_SIZE/3, SPACE_SIZE/2, SPACE_SIZE/2])

# First clump (250 particles)
particles[:NUM_PARTICLES//2, 0:3] = center1 + np.random.normal(0, 8, (NUM_PARTICLES//2, 3))
particles[:NUM_PARTICLES//2, 3:6] = np.random.normal(0, 0.5, (NUM_PARTICLES//2, 3))
particles[:NUM_PARTICLES//2, 6] = np.random.rand(NUM_PARTICLES//2) * 1.2 + 0.5
particles[:NUM_PARTICLES//2, 7] = np.random.choice([0,1,2], NUM_PARTICLES//2, p=[0.7,0.25,0.05])
particles[:NUM_PARTICLES//2, 8] = 1  # primary binary component

# Second clump (250 particles)
particles[NUM_PARTICLES//2:, 0:3] = center2 + np.random.normal(0, 8, (NUM_PARTICLES//2, 3))
particles[NUM_PARTICLES//2:, 3:6] = np.random.normal(0, 0.5, (NUM_PARTICLES//2, 3))
particles[NUM_PARTICLES//2:, 6] = np.random.rand(NUM_PARTICLES//2) * 1.2 + 0.5
particles[NUM_PARTICLES//2:, 7] = np.random.choice([0,1,2], NUM_PARTICLES//2, p=[0.7,0.25,0.05])
particles[NUM_PARTICLES//2:, 8] = 2  # secondary binary component

# Add initial orbital motion
binary_vec = center2 - center1
binary_dist = np.linalg.norm(binary_vec)
orbital_velocity = np.sqrt(G * NUM_PARTICLES * 0.8 / binary_dist) * 0.3

# Give initial velocities for orbital motion
particles[:NUM_PARTICLES//2, 3] += -orbital_velocity * binary_vec[1] / binary_dist
particles[:NUM_PARTICLES//2, 4] += orbital_velocity * binary_vec[0] / binary_dist
particles[NUM_PARTICLES//2:, 3] += orbital_velocity * binary_vec[1] / binary_dist
particles[NUM_PARTICLES//2:, 4] += -orbital_velocity * binary_vec[0] / binary_dist

# Color mapping
type_colors = {0: 'blue', 1: 'red', 2: 'gray'}

def get_colors(particles):
    colors = []
    for p in particles:
        if p[6] > FUSION_THRESHOLD:
            if p[8] == 1:
                colors.append('yellow')  # primary star ignited
            elif p[8] == 2:
                colors.append('orange')  # secondary star ignited
            else:
                colors.append('white')   # other ignited particle
        else:
            colors.append(type_colors[int(p[7])])
    return colors

# Calculate center of mass for each binary component
def get_binary_centers(particles):
    mask1 = (particles[:, 6] > 0) & (particles[:, 8] == 1)
    mask2 = (particles[:, 6] > 0) & (particles[:, 8] == 2)
    
    if np.any(mask1):
        com1 = np.average(particles[mask1, 0:3], weights=particles[mask1, 6], axis=0)
        total_mass1 = np.sum(particles[mask1, 6])
    else:
        com1 = np.array([0, 0, 0])
        total_mass1 = 0
        
    if np.any(mask2):
        com2 = np.average(particles[mask2, 0:3], weights=particles[mask2, 6], axis=0)
        total_mass2 = np.sum(particles[mask2, 6])
    else:
        com2 = np.array([0, 0, 0])
        total_mass2 = 0
        
    return com1, com2, total_mass1, total_mass2

# Update function with binary dynamics
def update_particles(particles):
    N = len(particles)
    
    # Get binary centers for tidal effects
    com1, com2, mass1, mass2 = get_binary_centers(particles)
    binary_sep = np.linalg.norm(com2 - com1) if mass1 > 0 and mass2 > 0 else 0
    
    # Particle-particle interactions
    for i in range(N):
        if particles[i, 6] <= 0:
            continue
        for j in range(i+1, N):
            if particles[j, 6] <= 0:
                continue
            
            # Distance vector
            dx, dy, dz = particles[j, 0:3] - particles[i, 0:3]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < 1e-5:
                dist = 1e-5
                
            # Gravity
            force = G * particles[i, 6] * particles[j, 6] / dist**2
            ax = force * dx / dist / particles[i, 6]
            ay = force * dy / dist / particles[i, 6]
            az = force * dz / dist / particles[i, 6]
            
            # Pressure when too close
            if dist < MERGE_DISTANCE * 3:
                ax -= PRESSURE_COEFF * dx
                ay -= PRESSURE_COEFF * dy
                az -= PRESSURE_COEFF * dz
                
            # Update velocities
            particles[i, 3:6] += np.array([ax, ay, az]) * TIME_STEP
            particles[j, 3:6] -= np.array([ax, ay, az]) * TIME_STEP
            
            # Merge if very close (only within same binary component)
            if dist < MERGE_DISTANCE and particles[i, 8] == particles[j, 8]:
                total_mass = particles[i, 6] + particles[j, 6]
                particles[i, 0:6] = (particles[i, 0:6] * particles[i, 6] + 
                                   particles[j, 0:6] * particles[j, 6]) / total_mass
                particles[i, 6] = total_mass
                particles[i, 7] = particles[i, 7] if particles[i, 6] >= particles[j, 6] else particles[j, 7]
                particles[j, 6] = 0  # remove particle j
    
    # Tidal interactions between binary components
    if binary_sep > 0 and binary_sep < BINARY_SEPARATION * 3:
        binary_dir = (com2 - com1) / binary_sep
        for p in particles:
            if p[6] <= 0:
                continue
            if p[8] == 1:  # primary component
                # Tidal force from secondary
                to_secondary = com2 - p[0:3]
                tidal_dist = np.linalg.norm(to_secondary)
                if tidal_dist > 1e-5:
                    tidal_force = TIDAL_FORCE_COEFF * mass2 / tidal_dist**2
                    p[3:6] += tidal_force * to_secondary / tidal_dist * TIME_STEP
            elif p[8] == 2:  # secondary component
                # Tidal force from primary
                to_primary = com1 - p[0:3]
                tidal_dist = np.linalg.norm(to_primary)
                if tidal_dist > 1e-5:
                    tidal_force = TIDAL_FORCE_COEFF * mass1 / tidal_dist**2
                    p[3:6] += tidal_force * to_primary / tidal_dist * TIME_STEP
    
    # Update positions
    particles[:, 0:3] += particles[:, 3:6] * TIME_STEP
    
    # Accretion and feedback
    for p in particles:
        if p[6] > 3 and np.random.rand() < PROB_GROWTH:
            p[6] += 0.12
            
        # Stellar wind feedback for ignited stars
        if p[6] > FUSION_THRESHOLD:
            for q in particles:
                if q[6] <= 0 or np.all(q == p):
                    continue
                vec = q[0:3] - p[0:3]
                dist = np.linalg.norm(vec)
                if dist < 8.0 and dist > 1e-5:
                    # Stronger feedback for different binary components
                    feedback = FEEDBACK_STRENGTH
                    if p[8] != q[8] and p[8] > 0 and q[8] > 0:
                        feedback *= 1.5
                    q[3:6] += feedback * vec / dist
    
    return particles

# Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
mask = particles[:, 6] > 0
colors = get_colors(particles[mask])
scat = ax.scatter(particles[mask, 0], particles[mask, 1], particles[mask, 2],
                  s=particles[mask, 6] * 6, c=colors, alpha=0.7)

ax.set_xlim(0, SPACE_SIZE)
ax.set_ylim(0, SPACE_SIZE)
ax.set_zlim(0, SPACE_SIZE)
ax.set_title("Binary Star Formation Simulation\n(Primary=yellow, Secondary=orange, H=blue, He=red, Dust=gray)")

def animate(frame):
    global particles
    particles = update_particles(particles)
    mask = particles[:, 6] > 0
    colors = get_colors(particles[mask])
    
    # Update scatter plot
    scat._offsets3d = (particles[mask, 0], particles[mask, 1], particles[mask, 2])
    scat.set_sizes(particles[mask, 6] * 6)
    scat.set_color(colors)
    
    # Update title with binary info
    com1, com2, mass1, mass2 = get_binary_centers(particles)
    if mass1 > 0 and mass2 > 0:
        binary_sep = np.linalg.norm(com2 - com1)
        ax.set_title(f"Binary Star Formation - Separation: {binary_sep:.1f} units\n"
                    f"Primary Mass: {mass1:.1f}, Secondary Mass: {mass2:.1f}")
    
    return scat,

ani = animation.FuncAnimation(fig, animate, frames=400, interval=40, blit=False)
plt.show()
