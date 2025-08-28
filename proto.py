import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
NUM_PARTICLES = 200
SPACE_SIZE = 100
G = 1.0
TIME_STEP = 0.1
MERGE_DISTANCE = 2.0
PROB_GROWTH = 0.1
FUSION_THRESHOLD = 5.0  # mass at which protostar ignites

# Initialize particles
# Each particle: [x, y, vx, vy, mass, type]
# type: 0 = hydrogen, 1 = helium, 2 = dust
particles = np.zeros((NUM_PARTICLES, 6))
particles[:, 0:2] = np.random.rand(NUM_PARTICLES, 2) * SPACE_SIZE
particles[:, 2:4] = (np.random.rand(NUM_PARTICLES, 2) - 0.5)
particles[:, 4] = np.random.rand(NUM_PARTICLES) * 1 + 0.5
particles[:, 5] = np.random.choice([0,1,2], NUM_PARTICLES, p=[0.7,0.25,0.05])

# Colors for plotting
type_colors = {0: 'blue', 1: 'red', 2: 'gray'}

# Update function
def update_particles(particles):
    N = len(particles)
    for i in range(N):
        if particles[i,4] <= 0:
            continue
        for j in range(i+1, N):
            if particles[j,4] <= 0:
                continue
            dx = particles[j,0] - particles[i,0]
            dy = particles[j,1] - particles[i,1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-5:
                dist = 1e-5
            force = G * particles[i,4] * particles[j,4] / dist**2
            ax = force * dx / dist / particles[i,4]
            ay = force * dy / dist / particles[i,4]
            particles[i,2] += ax * TIME_STEP
            particles[i,3] += ay * TIME_STEP
            particles[j,2] -= ax * TIME_STEP
            particles[j,3] -= ay * TIME_STEP
            if dist < MERGE_DISTANCE:
                total_mass = particles[i,4] + particles[j,4]
                particles[i,0:4] = (particles[i,0:4]*particles[i,4] + particles[j,0:4]*particles[j,4]) / total_mass
                particles[i,4] = total_mass
                particles[i,5] = particles[i,5] if particles[i,4] >= particles[j,4] else particles[j,5]
                particles[j,4] = 0
    particles[:,0] += particles[:,2]*TIME_STEP
    particles[:,1] += particles[:,3]*TIME_STEP

    for p in particles:
        if p[4] > 2 and np.random.rand() < PROB_GROWTH:
            p[4] += 0.1
    
    return particles

# Visualization
fig, ax = plt.subplots()
mask = particles[:,4] > 0

def get_colors(particles):
    colors = []
    for p in particles:
        if p[4] > FUSION_THRESHOLD:
            colors.append('yellow')  # protostar ignites
        else:
            colors.append(type_colors[p[5]])
    return colors

colors = get_colors(particles[mask])
scat = ax.scatter(particles[mask,0], particles[mask,1], s=particles[mask,4]*10, c=colors)

def animate(frame):
    global particles
    particles = update_particles(particles)
    mask = particles[:,4] > 0
    colors = get_colors(particles[mask])
    scat.set_offsets(particles[mask,0:2])
    scat.set_sizes(particles[mask,4]*10)
    scat.set_color(colors)
    return scat,

ax.set_xlim(0, SPACE_SIZE)
ax.set_ylim(0, SPACE_SIZE)
ax.set_title("Protostar Formation Simulation (H=blue, He=red, Dust=gray, Ignited=yellow)")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()

