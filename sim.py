import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Simulation Parameters
NUM_PARTICLES = 300
SPACE_SIZE = 100
G = 1.0
TIME_STEP = 0.05
MERGE_DISTANCE = 2.0
PROB_GROWTH = 0.1
FUSION_THRESHOLD = 5.0
ROTATION_SPEED = 0.01
PRESSURE_COEFF = 0.05
FEEDBACK_STRENGTH = 0.05

# Initialize Particles
# Each particle: [x, y, z, vx, vy, vz, mass, type]
# type: 0 = hydrogen, 1 = helium, 2 = dust
particles = np.zeros((NUM_PARTICLES, 8))
particles[:,0:3] = np.random.rand(NUM_PARTICLES,3) * SPACE_SIZE
particles[:,3:6] = (np.random.rand(NUM_PARTICLES,3) - 0.5)
particles[:,6] = np.random.rand(NUM_PARTICLES)*1 + 0.5
particles[:,7] = np.random.choice([0,1,2], NUM_PARTICLES, p=[0.7,0.25,0.05])

# Add initial rotation around Z-axis
center = SPACE_SIZE / 2
r = particles[:,0:3] - center
particles[:,3] += -ROTATION_SPEED * (r[:,1])
particles[:,4] += ROTATION_SPEED * (r[:,0])

# Color mapping
type_colors = {0: 'blue', 1: 'red', 2: 'gray'}

def get_colors(particles):
    colors = []
    for p in particles:
        if p[6] > FUSION_THRESHOLD:
            colors.append('yellow')
        else:
            colors.append(type_colors[p[7]])
    return colors

# Update function
def update_particles(particles):
    N = len(particles)
    for i in range(N):
        if particles[i,6] <= 0:
            continue
        for j in range(i+1,N):
            if particles[j,6] <= 0:
                continue
            # distance vector
            dx,dy,dz = particles[j,0:3] - particles[i,0:3]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < 1e-5:
                dist = 1e-5
            # gravity
            force = G * particles[i,6] * particles[j,6] / dist**2
            ax = force * dx / dist / particles[i,6]
            ay = force * dy / dist / particles[i,6]
            az = force * dz / dist / particles[i,6]
            # simple repulsive pressure when too close
            if dist < MERGE_DISTANCE*3:
                ax -= PRESSURE_COEFF * dx
                ay -= PRESSURE_COEFF * dy
                az -= PRESSURE_COEFF * dz
            # update velocities
            particles[i,3:6] += np.array([ax,ay,az]) * TIME_STEP
            particles[j,3:6] -= np.array([ax,ay,az]) * TIME_STEP
            # merge if very close
            if dist < MERGE_DISTANCE:
                total_mass = particles[i,6] + particles[j,6]
                particles[i,0:6] = (particles[i,0:6]*particles[i,6] + particles[j,0:6]*particles[j,6]) / total_mass
                particles[i,6] = total_mass
                # inherit type of more massive particle
                particles[i,7] = particles[i,7] if particles[i,6]>=particles[j,6] else particles[j,7]
                particles[j,6] = 0  # remove particle j

    # update positions
    particles[:,0:3] += particles[:,3:6]*TIME_STEP

    # accretion & feedback
    for p in particles:
        if p[6] > 2 and np.random.rand() < PROB_GROWTH:
            p[6] += 0.1
        # simple feedback: push nearby particles away
        if p[6] > FUSION_THRESHOLD:
            for q in particles:
                if q[6] <= 0 or np.all(q==p):
                    continue
                vec = q[0:3]-p[0:3]
                dist = np.linalg.norm(vec)
                if dist < 5.0 and dist>1e-5:
                    q[3:6] += FEEDBACK_STRENGTH * vec/dist

    return particles

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mask = particles[:,6] > 0
colors = get_colors(particles[mask])
scat = ax.scatter(particles[mask,0], particles[mask,1], particles[mask,2],
                  s=particles[mask,6]*10, c=colors)

ax.set_xlim(0, SPACE_SIZE)
ax.set_ylim(0, SPACE_SIZE)
ax.set_zlim(0, SPACE_SIZE)
ax.set_title("3D Protostar Formation Simulation")

def animate(frame):
    global particles
    particles = update_particles(particles)
    mask = particles[:,6] > 0
    colors = get_colors(particles[mask])
    scat._offsets3d = (particles[mask,0], particles[mask,1], particles[mask,2])
    scat.set_sizes(particles[mask,6]*10)
    scat.set_color(colors)
    return scat,

ani = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=False)
plt.show()

