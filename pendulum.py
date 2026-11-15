import math
import matplotlib.pyplot as plt
import numpy as np

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)



class SuperQueens:

    def __init__(self, n):
        self.n = n
        c1 = math.pow(phi, 2/math.pi)
        a = [(math.pi*2*i)/n for i in range(n)]
        self.r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]
        self.intersection_points = []
        self.intersection_points_cartesian = []
        for i in range(self.n):
            angle = (2 * np.pi * i) / self.n
            for r in self.r:
                self.intersection_points.append((angle, r))
                # Convert to Cartesian coordinates
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                self.intersection_points_cartesian.append((x, y))
        
    
    def draw(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, 2*np.pi, 100)
        for r in self.r:
            radius = np.full_like(theta, r)
            ax.plot(theta, radius)
        
        # Draw lines from origin to each circle at 360/n intervals
        for i in range(self.n):
            angle = (2 * np.pi * i) / self.n  # Convert to radians
            # Draw line from origin (0,0) to the largest radius
            ax.plot([angle, angle], [0, max(self.r)], 'r-', linewidth=2)
        
        # Plot intersection points
        for angle, r in self.intersection_points:
            ax.plot(angle, r, 'ko', markersize=8)  # Black dots for intersection points
        
        plt.show()


class MultiplePendulums:
    def __init__(self, n_pendulums, lengths=None, masses=None, initial_angles=None, initial_velocities=None, g=9.81):
        """
        Initialize multiple pendulums simulation
        
        Parameters:
        - n_pendulums: number of pendulums
        - lengths: list of pendulum lengths (default: all 1.0)
        - masses: list of pendulum masses (default: all 1.0)
        - initial_angles: list of initial angles in radians (default: all π/4)
        - initial_velocities: list of initial angular velocities (default: all 0.0)
        - g: gravitational acceleration (default: 9.81 m/s²)
        """
        self.n_pendulums = n_pendulums
        self.g = g
        
        # Set default values if not provided
        self.lengths = lengths if lengths is not None else [1.0] * n_pendulums
        self.masses = masses if masses is not None else [1.0] * n_pendulums
        self.initial_angles = initial_angles if initial_angles is not None else [np.pi/4] * n_pendulums
        self.initial_velocities = initial_velocities if initial_velocities is not None else [0.0] * n_pendulums
        
        # Validate input lengths
        if len(self.lengths) != n_pendulums or len(self.masses) != n_pendulums or \
           len(self.initial_angles) != n_pendulums or len(self.initial_velocities) != n_pendulums:
            raise ValueError("All parameter lists must have length equal to n_pendulums")
    
    def derivatives(self, state, t):
        """
        Calculate derivatives for the system of pendulums
        state: [θ1, θ2, ..., θn, ω1, ω2, ..., ωn]
        """
        angles = state[:self.n_pendulums]
        velocities = state[self.n_pendulums:]
        
        # Calculate angular accelerations
        accelerations = []
        for i in range(self.n_pendulums):
            # Simple pendulum equation: d²θ/dt² = -(g/L) * sin(θ)
            acc = -(self.g / self.lengths[i]) * np.sin(angles[i])
            accelerations.append(acc)
        
        return np.concatenate([velocities, accelerations])
    
    def simulate(self, t_span, dt=0.01):
        """
        Simulate the pendulums for given time span
        
        Parameters:
        - t_span: tuple (t_start, t_end) or single value for t_end
        - dt: time step for simulation
        
        Returns:
        - t: time array
        - angles: array of angles for each pendulum over time
        - velocities: array of velocities for each pendulum over time
        """
        from scipy.integrate import odeint
        
        if isinstance(t_span, (int, float)):
            t_span = (0, t_span)
        
        # Create time array
        t = np.arange(t_span[0], t_span[1], dt)
        
        # Initial state: [θ1, θ2, ..., θn, ω1, ω2, ..., ωn]
        initial_state = np.array(self.initial_angles + self.initial_velocities)
        
        # Solve ODE using odeint
        solution = odeint(self.derivatives, initial_state, t)
        
        # Separate angles and velocities
        angles = solution[:, :self.n_pendulums].T
        velocities = solution[:, self.n_pendulums:].T
        
        return t, angles, velocities
    
    def animate(self, t_span, dt=0.01, save_animation=False, filename='pendulum_animation.gif'):
        """
        Animate the pendulum simulation
        """
        t, angles, velocities = self.simulate(t_span, dt)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-max(self.lengths)*1.2, max(self.lengths)*1.2)
        ax.set_ylim(-max(self.lengths)*1.2, max(self.lengths)*1.2)
        ax.set_ylim(-max(self.lengths)*1.2, max(self.lengths)*1.2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Multiple Pendulums Simulation')
        
        # Create pendulum lines and bobs
        lines = []
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_pendulums))
        
        for i in range(self.n_pendulums):
            line, = ax.plot([], [], 'o-', color=colors[i], linewidth=2, markersize=8, 
                          label=f'Pendulum {i+1} (L={self.lengths[i]:.1f}, m={self.masses[i]:.1f})')
            lines.append(line)
        
        ax.legend()
        
        def animate(frame):
            for i in range(self.n_pendulums):
                # Calculate pendulum position
                x = self.lengths[i] * np.sin(angles[i, frame])
                y = -self.lengths[i] * np.cos(angles[i, frame])
                
                # Update line from origin to bob
                lines[i].set_data([0, x], [0, y])
            
            return lines
        
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
        
        if save_animation:
            anim.save(filename, writer='pillow', fps=20)
        
        plt.show()
        return anim
    
    def plot_trajectories(self, t_span, dt=0.01):
        """
        Plot the angular trajectories of all pendulums
        """
        t, angles, velocities = self.simulate(t_span, dt)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_pendulums))
        
        # Plot angles
        for i in range(self.n_pendulums):
            ax1.plot(t, angles[i], color=colors[i], label=f'Pendulum {i+1}')
        ax1.set_ylabel('Angle (radians)')
        ax1.set_title('Angular Displacement vs Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot velocities
        for i in range(self.n_pendulums):
            ax2.plot(t, velocities[i], color=colors[i], label=f'Pendulum {i+1}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Angular Velocity vs Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


class PendulumGeneticAlgorithm:
    def __init__(self, n_pendulums, intersection_points_cartesian, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        """
        Genetic Algorithm for optimizing pendulum configurations
        
        Parameters:
        - n_pendulums: number of pendulums
        - intersection_points_cartesian: list of (x, y) tuples from SuperQueens
        - population_size: size of the population
        - mutation_rate: probability of mutation
        - crossover_rate: probability of crossover
        """
        self.n_pendulums = n_pendulums
        self.intersection_points_cartesian = intersection_points_cartesian
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Parameter bounds for realistic pendulum configurations
        self.length_bounds = (0.0, 1.0)  # Pendulum lengths between 0 and 1 to reach points within unit circle
        self.mass_bounds = (0.1, 2.0)
        self.angle_bounds = (-np.pi, np.pi)
        self.velocity_bounds = (-2.0, 2.0)
        
        # Initialize population
        self.population = self.initialize_population()
        self.best_fitness = 0
        self.best_individual = None
    
    def initialize_population(self):
        """Initialize random population of pendulum configurations"""
        population = []
        for _ in range(self.population_size):
            individual = {
                'lengths': np.random.uniform(self.length_bounds[0], self.length_bounds[1], self.n_pendulums),
                'masses': np.random.uniform(self.mass_bounds[0], self.mass_bounds[1], self.n_pendulums),
                'initial_angles': np.random.uniform(self.angle_bounds[0], self.angle_bounds[1], self.n_pendulums),
                'initial_velocities': np.random.uniform(self.velocity_bounds[0], self.velocity_bounds[1], self.n_pendulums)
            }
            population.append(individual)
        return population
    
    def fitness_function(self, individual):
        """
        Calculate fitness based on how close the pendulum gets to different radii in order
        """
        try:
            # Create pendulum system
            pendulum = MultiplePendulums(
                n_pendulums=self.n_pendulums,
                lengths=individual['lengths'],
                masses=individual['masses'],
                initial_angles=individual['initial_angles'],
                initial_velocities=individual['initial_velocities']
            )
            
            # Simulate for 10π seconds
            t_span = 10 * np.pi
            dt = 0.01
            t, angles, velocities = pendulum.simulate(t_span, dt)
            
            # Get the last pendulum's trajectory
            last_pendulum_angles = angles[-1]  # Last pendulum
            
            # Calculate positions and radii of the last pendulum
            positions = []
            radii = []
            for i, angle in enumerate(last_pendulum_angles):
                x = individual['lengths'][-1] * np.sin(angle)
                y = -individual['lengths'][-1] * np.cos(angle)
                positions.append((x, y))
                # Calculate radius from origin
                radius = np.sqrt(x**2 + y**2)
                radii.append(radius)
            
            # Get unique target radii from intersection points (sorted)
            target_radii = sorted(list(set([np.sqrt(x**2 + y**2) for x, y in self.intersection_points_cartesian])))
            
            # Calculate fitness based on proximity to target radii
            total_fitness = 0
            radius_tolerance = 0.1  # Tolerance for considering "close" to a radius
            
            for target_radius in target_radii:
                # Find minimum distance to this target radius
                min_distance = float('inf')
                for radius in radii:
                    distance = abs(radius - target_radius)
                    if distance < min_distance:
                        min_distance = distance
                
                # Convert distance to fitness (closer = higher fitness)
                if min_distance <= radius_tolerance:
                    # Bonus for being very close
                    radius_fitness = 10.0 * (1.0 - min_distance / radius_tolerance)
                else:
                    # Penalty for being far, but still some fitness
                    radius_fitness = max(0, 1.0 - min_distance)
                
                total_fitness += radius_fitness
            
            # Bonus for visiting multiple different radii
            visited_radii = set()
            for radius in radii:
                for target_radius in target_radii:
                    if abs(radius - target_radius) <= radius_tolerance:
                        visited_radii.add(target_radius)
                        break
            
            diversity_bonus = len(visited_radii) * 5.0
            
            return total_fitness + diversity_bonus
            
        except Exception as e:
            # Return low fitness for invalid configurations
            return 0
    
    def selection(self, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            # Random tournament
            tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = {}
        child2 = {}
        
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child1[key] = parent1[key].copy()
                child2[key] = parent2[key].copy()
            else:
                child1[key] = parent2[key].copy()
                child2[key] = parent1[key].copy()
        
        return child1, child2
    
    def mutation(self, individual):
        """Gaussian mutation"""
        for key in individual.keys():
            if np.random.random() < self.mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1, individual[key].shape)
                individual[key] += noise
                
                # Clip to bounds
                if key == 'lengths':
                    individual[key] = np.clip(individual[key], self.length_bounds[0], self.length_bounds[1])
                elif key == 'masses':
                    individual[key] = np.clip(individual[key], self.mass_bounds[0], self.mass_bounds[1])
                elif key == 'initial_angles':
                    individual[key] = np.clip(individual[key], self.angle_bounds[0], self.angle_bounds[1])
                elif key == 'initial_velocities':
                    individual[key] = np.clip(individual[key], self.velocity_bounds[0], self.velocity_bounds[1])
    
    def evolve(self, generations=100):
        """Run the genetic algorithm"""
        best_fitness_history = []
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.fitness_function(ind) for ind in self.population]
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            best_fitness_history.append(self.best_fitness)
            
            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness}")
            
            # Selection
            selected = self.selection(fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(self.population[best_idx].copy())
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                self.mutation(child1)
                self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            self.population = new_population[:self.population_size]
        
        return best_fitness_history
    
    def get_best_pendulum(self):
        """Return the best pendulum configuration as a MultiplePendulums object"""
        if self.best_individual is None:
            return None
        
        return MultiplePendulums(
            n_pendulums=self.n_pendulums,
            lengths=self.best_individual['lengths'],
            masses=self.best_individual['masses'],
            initial_angles=self.best_individual['initial_angles'],
            initial_velocities=self.best_individual['initial_velocities']
        )


# Example usage
if __name__ == "__main__":
    # Create intersection points from SuperQueens
    super_queens = SuperQueens(6)
    intersection_points_cartesian = super_queens.intersection_points_cartesian
    
    # Create and run genetic algorithm
    ga = PendulumGeneticAlgorithm(
        n_pendulums=2,
        intersection_points_cartesian=intersection_points_cartesian,
        population_size=1000,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    print("Starting Genetic Algorithm...")
    fitness_history = ga.evolve(generations=50)
    
    print(f"\nBest fitness achieved: {ga.best_fitness}")
    print("Best individual parameters:")
    if ga.best_individual:
        for key, value in ga.best_individual.items():
            print(f"  {key}: {value}")
    
    # Visualize the best pendulum
    best_pendulum = ga.get_best_pendulum()
    if best_pendulum:
        best_pendulum.plot_trajectories(t_span=10*np.pi, dt=0.01)
        # best_pendulum.animate(t_span=10*np.pi, dt=0.01)
    
    
