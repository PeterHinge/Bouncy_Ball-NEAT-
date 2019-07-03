import math
import random
import pygame

########################################################################################################################
SCREENSIZE = 1000, 600


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


########################################################################################################################


class Neuron:
    def __init__(self, place_in_network):
        self.no = place_in_network  # represented as [layer, index_in_layer]. Especially useful when keeping track of innovation connections
        self.connections = []  # connected forward neurons, useful
        self.value = 0  # accumulated inputs (prior to activation)
        self.activated_value = 0  # value post activation function
        self.enabled = True

    def sigmoid_activation(self):
        self.activated_value = 1 / (1 + math.exp(-self.value))


class Synapse:
    def __init__(self, inno_no, from_neuron, to_neuron, weight):
        self.inno_no = inno_no  # Innovation number to keep track when performing encoding and crossover
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        self.enabled = True


class NeuralNet:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.hidden = []
        self.fitness = 0

        self.neurons = []  # neurons
        self.synapses = []  # synapses
        self.layers = []  # layers

    def generate_empty_genome(self, input_size, output_size):
        self.neurons = []
        self.synapses = []
        self.layers = []
        self.hidden = []
        self.fitness = 0

        self.inputs = [Neuron(['input', i]) for i in range(input_size)]
        self.outputs = [Neuron(['output', i]) for i in range(output_size)]

        self.neurons += self.inputs + self.outputs
        self.layers.append(self.inputs)
        self.layers.append(self.outputs)

    def generate_synapse(self, inno_no, from_neuron, to_neuron, weight=random.uniform(-2, 2)):
        from_neuron.connections.append(to_neuron)

        connection = Synapse(inno_no, from_neuron, to_neuron, weight)
        self.synapses.append(connection)

    def mutate_synapse(self, shift=True):
        if len(self.synapses) != 0:
            synapse = self.synapses[random.randint(0, len(self.synapses) - 1)]

            # 2 possibilities: a randomized (new) or adjusted (shifted) value
            if shift:
                synapse.weight = random.uniform(0, 2)
            else:
                synapse.weight * random.uniform(-2, 2)

    def enable_disable_synapse(self):
        if len(self.synapses) != 0:
            synapse = self.synapses[random.randint(0, len(self.synapses) - 1)]
            if synapse.enabled:
                synapse.enabled = False
            elif not synapse.enabled:
                synapse.enabled = True

    def generate_new_neuron(self):
        neuron = Neuron(['hidden', len(self.hidden)])
        self.hidden.append(neuron)
        self.neurons.append(neuron)

    def enable_disable_neuron(self):
        if len(self.hidden) != 0:
            neuron = self.hidden[random.randint(0, len(self.hidden) - 1)]
            if neuron.enabled:
                neuron.enabled = False
            elif not neuron.enabled:
                neuron.enabled = True


class NEAT:
    def __init__(self, max_gen=None):
        self.population = []  # a list of all the current networks
        self.species = {}  # a dictionary where the keys are strings of a list of inno_nums indicating the species and where the values are networks belonging to that species
        self.inno_connections = []
        self.size = 0
        self.current_gen = 1
        self.max_gen = max_gen

    def generate_new_population(self, size):
        self.current_gen = 1

        self.population = [NeuralNet() for _ in range(size)]
        self.size = size

    def perform_mutation(self):
        for network in self.population:

            # 10% chance to spawn a random synapse (100% on first generation)
            if random.randint(0, 9) == 0:
                random_neuron_1 = network.neurons[random.randint(0, len(network.neurons) - 1)]
                random_neuron_2 = None
                while random_neuron_2 is None:
                    current_neuron = network.neurons[random.randint(0, len(network.neurons) - 1)]
                    if current_neuron.no[0] == random_neuron_1.no[0] and current_neuron.no[0] != 'hidden':
                        continue
                    else:
                        random_neuron_2 = current_neuron

                if random_neuron_1.no[0] == 'output' or random_neuron_2.no[0] == 'input':
                    if [random_neuron_2.no, random_neuron_1.no] not in self.inno_connections:
                        self.inno_connections.append([random_neuron_2.no, random_neuron_1.no])

                    inno_num = self.inno_connections.index([random_neuron_2.no, random_neuron_1.no])
                    if inno_num not in [i.inno_no for i in network.synapses]:
                        network.generate_synapse(inno_num, random_neuron_2, random_neuron_1, random.uniform(-2, 2))
                        print("Synapse generated in network {} between {} and {}.".format(str(self.population.index(network)), str(random_neuron_2.no), str(random_neuron_1.no)))

                    else:  # if there's already such a connection, a mutation will happen instead
                        network.mutate_synapse()

                else:
                    if [random_neuron_1.no, random_neuron_2.no] not in self.inno_connections:
                        self.inno_connections.append([random_neuron_1.no, random_neuron_2.no])
                    inno_num = self.inno_connections.index([random_neuron_1.no, random_neuron_2.no])
                    if inno_num not in [i.inno_no for i in network.synapses]:
                        network.generate_synapse(inno_num, random_neuron_1, random_neuron_2, random.uniform(-2, 2))
                        print("Synapse generated in network {} between {} and {}.".format(str(self.population.index(network)), str(random_neuron_1.no), str(random_neuron_2.no)))

                    else:  # if there's already such a connection, a mutation will happen instead
                        network.mutate_synapse()

            # 5% chance to mutate a random synapse (if one exists)
            elif random.randint(0, 19) == 0:
                network.mutate_synapse()

            # 5% chance to spawn a randomly connected new neuron
            elif random.randint(0, 19) == 0:
                new_neuron = Neuron(['hidden', len(network.hidden)])
                random_neuron_1 = network.neurons[random.randint(0, len(network.neurons) - 1)]
                random_neuron_2 = None
                while random_neuron_2 is None:
                    current_neuron = network.neurons[random.randint(0, len(network.neurons) - 1)]
                    if current_neuron.no[0] == random_neuron_1.no[0] and current_neuron.no[0] != 'hidden':
                        continue
                    else:
                        random_neuron_2 = current_neuron

                if random_neuron_1.no[0] == 'output' or random_neuron_2.no[0] == 'input':
                    if [random_neuron_2.no, new_neuron.no] not in self.inno_connections:
                        self.inno_connections.append([random_neuron_2.no, new_neuron.no])
                    if [new_neuron.no, random_neuron_1.no] not in self.inno_connections:
                        self.inno_connections.append([new_neuron.no, random_neuron_1.no])

                    inno_num_1 = self.inno_connections.index([random_neuron_2.no, new_neuron.no])
                    inno_num_2 = self.inno_connections.index([new_neuron.no, random_neuron_1.no])

                    network.generate_synapse(inno_num_1, random_neuron_2, new_neuron, 2)
                    network.generate_synapse(inno_num_2, new_neuron, random_neuron_1, random.uniform(-2, 2))

                    print("Neuron generated in network {} with connections between {} and {}.".format(str(self.population.index(network)), str(random_neuron_2.no), str(random_neuron_1.no)))

                else:
                    if [random_neuron_1.no, new_neuron.no] not in self.inno_connections:
                        self.inno_connections.append([random_neuron_1.no, new_neuron.no])
                    if [new_neuron.no, random_neuron_2.no] not in self.inno_connections:
                        self.inno_connections.append([new_neuron.no, random_neuron_2.no])

                    inno_num_1 = self.inno_connections.index([random_neuron_1.no, new_neuron.no])
                    inno_num_2 = self.inno_connections.index([new_neuron.no, random_neuron_2.no])

                    network.generate_synapse(inno_num_1, random_neuron_1, new_neuron, 2)
                    network.generate_synapse(inno_num_2, new_neuron, random_neuron_2, random.uniform(-2, 2))

                    print("Neuron generated in network {} with connections between {} and {}.".format(str(self.population.index(network)), str(random_neuron_1.no), str(random_neuron_2.no)))

                network.hidden.append(new_neuron)
                network.neurons.append(new_neuron)

    # makes species, then sorts the species and removes the least fit networks, then breeds and mutates new generation
    def perform_crossover(self):

        self.perform_speciation()

        for networks in self.species.values():
            if len(networks) == 1:
                continue
            else:
                networks.sort(key=lambda x: x.fitness, reverse=True)
                survivors = len(networks) // 2 + 1
                while len(networks) != survivors:
                    network_to_remove = networks[-1]
                    networks.remove(network_to_remove)
                    self.population.remove(network_to_remove)

        print("Population size after selection: {}".format(len(self.population)))
        print("Cloning, breeding and repopulating...")

        self.perform_breeding()

        self.current_gen += 1

        self.perform_mutation()

    # divides self.population into self.species based on their network structure (does not sort for fitness)
    def perform_speciation(self):
        self.species = {}  # last generation's species cleared
        for network in self.population:
            gene = []
            for synapse in network.synapses:
                gene.append(synapse.inno_no)
            gene.sort()
            if str(gene) in self.species:
                self.species[str(gene)].append(network)
            else:
                self.species.setdefault(str(gene), [network])

    # based on species and the individual networks within these, different types of breeding is performed
    def perform_breeding(self):
        species_fitness = []
        for species, networks in self.species.items():
            if len(networks) >= 3:
                fit = (networks[0].fitness + networks[1].fitness + networks[2].fitness) // 3
                species_fitness.append([species, fit])
            else:
                species_fitness.append([species, networks[0].fitness])

        species_fitness.sort(key=lambda x: x[1], reverse=True)

        while len(self.population) != self.size:
            chance = random.randint(0, 9)
            parent_1 = None
            parent_2 = None

            same_species = False

            # 40% chance for to clone top network in a species
            if chance <= 3:
                while parent_1 is None:
                    for i in range(len(self.species)):
                        if random.randint(0, 1) == 0:
                            parent_1 = self.species[species_fitness[i][0]][0]
                            print("Cloned top from species {}.".format(species_fitness[i][0]))
                            break

            # 40% chance for top species to breed with itself
            elif chance <= 7 and len(self.species[species_fitness[0][0]]) >= 2:
                parent_1 = self.species[species_fitness[0][0]][0]
                parent_2 = self.species[species_fitness[0][0]][1]
                same_species = True
                print("breeded from top species {}.".format(species_fitness[0][0]))

            # 20% chance for top to species to breed with each other
            elif chance <= 9:
                parent_1 = self.species[species_fitness[0][0]][0]
                parent_2 = self.species[species_fitness[1][0]][0]
                print("breeded from top 2 species {} and {}.".format(species_fitness[0][0], species_fitness[1][0]))

            # cloning
            new_network = NeuralNet()
            new_network.generate_empty_genome(len(parent_1.inputs), len(parent_1.outputs))
            num_of_new_hidden = len(parent_1.hidden) - len(new_network.hidden)
            while num_of_new_hidden > 0:
                new_network.generate_new_neuron()
                num_of_new_hidden -= 1

            for i, synapse in enumerate(parent_1.synapses):
                inno_num = synapse.inno_no

                going_from = new_network.neurons[parent_1.neurons.index(synapse.from_neuron)]
                going_to = new_network.neurons[parent_1.neurons.index(synapse.to_neuron)]
                if same_species:
                    w = random.uniform(synapse.weight, parent_2.synapses[i].weight)
                else:
                    w = synapse.weight
                new_network.synapses.append(Synapse(inno_num, going_from, going_to, w))
                going_from.connections.append(going_to)

            # breeding
            if parent_2 is not None and not same_species:

                num_of_new_hidden = len(parent_2.hidden) - len(new_network.hidden)
                while num_of_new_hidden > 0:
                    new_network.generate_new_neuron()
                    num_of_new_hidden -= 1

                for i, synapse in enumerate(parent_2.synapses):
                    inno_num = synapse.inno_no
                    going_from = new_network.neurons[parent_2.neurons.index(synapse.from_neuron)]
                    going_to = new_network.neurons[parent_2.neurons.index(synapse.to_neuron)]
                    w = synapse.weight
                    if going_to not in going_from.connections:
                        new_network.synapses.append(Synapse(inno_num, going_from, going_to, w))
                        going_from.connections.append(going_to)

            self.population.append(new_network)


########################################################################################################################


class Ball:
    def __init__(self, color, x, y, size=20, speed=0):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.speed = speed  # Balls vertical speed
        self.fitness = 0  # How many pixels the ball travels before death

    def apply_gravity(self):
        if self.speed <= 5:
            self.speed += 1

        self.y += self.speed

    def out_of_bounds(self):
        return 0 >= self.y or self.y + self.size >= SCREENSIZE[1]


class PipePair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = GREEN
        self.size = 70
        self.speed = 2  # Pipe pairs horizontal speed
        self.space = 150  # Space between pipe pairs

    def move(self):
        self.x -= self.speed


def display_score(current_score, screen):
    font = pygame.font.SysFont(None, 60)
    text = font.render("Score: {}".format(current_score), True, WHITE)
    screen.blit(text, [0, 0])


def display_gen(current_gen, screen):
    font = pygame.font.SysFont(None, 60)
    text = font.render("Gen: {}".format(current_gen), True, WHITE)
    screen.blit(text, [0, 560])


def display_top_net(network, screen):
    fitness, net = network

    for i in range(len(net.inputs)):
        pygame.draw.circle(screen, WHITE, [700, 570 - (i*30)], 10)

    pygame.draw.circle(screen, WHITE, [900, 530], 10)

    for i in range(len(net.hidden)):
        pygame.draw.circle(screen, WHITE, [800, 570 - (i * 30)], 10)

    for synapse in net.synapses:
        if synapse.weight > 0:
            color = GREEN
        else:
            color = RED
        if synapse.from_neuron.no[0] == 'input':
            start = [700, 570 - (synapse.from_neuron.no[1] * 30)]
        else:
            start = [800, 570 - (synapse.from_neuron.no[1] * 30)]
        if synapse.to_neuron.no[0] == 'output':
            end = [900, 530]
        else:
            end = [800, 570 - (synapse.to_neuron.no[1] * 30)]
        pygame.draw.line(screen, color, start, end, 5)

def collision(object_1, object_2):
    return 220 >= object_2.x >= 130 and (object_2.y + object_2.space <= object_1.y + 20 or object_1.y <= object_2.y)


def bouncy_ball(neural_nets, current_gen):
    pygame.init()
    pygame.display.set_caption('Bouncy Ball')

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREENSIZE)
    score = 0

    balls = [Ball(YELLOW, 200, 300) for _ in range(len(neural_nets))]
    dead_balls = 0
    top_network = 0, neural_nets[0]
    pipe_pairs = []

    exit_program = False

    while not exit_program:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:  # Quit game event
                exit_program = True

        # Logic section
        if len(balls) == dead_balls:
            exit_program = True

        if len(pipe_pairs) == 0 or pipe_pairs[-1].x == 500:
            pipe_pairs.append(PipePair(SCREENSIZE[0], random.randint(50, 400)))

        if pipe_pairs[0].x == -70:
            pipe_pairs.pop(0)

        elif pipe_pairs[0].x == 150:
            score += 1

        for i, network in enumerate(neural_nets):
            input_neuron_0, input_neuron_1, input_neuron_2, input_neuron_3 = network.inputs
            input_neuron_0.value = pipe_pairs[0].x - (balls[i].x + balls[i].size + 1)  # distance to next pipe pair
            input_neuron_1.value = pipe_pairs[0].y - balls[i].y  # distance to top pipe
            input_neuron_2.value = (balls[i].y + balls[i].size + 1) - (pipe_pairs[0].y + pipe_pairs[0].space)  # bottom pipe
            input_neuron_3.value = balls[i].speed  # the balls vertical speed

            for synapse in network.synapses:
                synapse.from_neuron.sigmoid_activation()
                synapse.to_neuron.value = synapse.from_neuron.activated_value * synapse.weight

            for neuron in network.outputs:
                neuron.sigmoid_activation()

                if neuron.activated_value >= 0.7:
                    balls[i].speed = -10

        # Drawing section
        screen.fill(BLACK)

        for i, ball in enumerate(balls):
            if ball.color != RED:

                if ball.out_of_bounds() or collision(ball, pipe_pairs[0]):
                    ball.color = RED
                    dead_balls += 1
                else:
                    pygame.draw.circle(screen, ball.color, [ball.x, ball.y], ball.size)
                    ball.apply_gravity()
                    ball.fitness += 1

                if ball.fitness > top_network[0]:
                    top_network = ball.fitness, neural_nets[i]

        for pipe_pair in pipe_pairs:
            pygame.draw.rect(screen, GREEN, [pipe_pair.x, 0, pipe_pair.size, pipe_pair.y])
            pygame.draw.rect(screen, GREEN, [pipe_pair.x, pipe_pair.y + pipe_pair.space, pipe_pair.size, SCREENSIZE[1]])
            pipe_pair.move()

        display_score(score, screen)
        display_gen(current_gen, screen)
        display_top_net(top_network, screen)

        pygame.display.flip()
        clock.tick(60)

    for i, network in enumerate(neural_nets):
        network.fitness = balls[i].fitness

    pygame.display.quit()


if __name__ == '__main__':
    neat = NEAT()
    neat.generate_new_population(500)

    for net in neat.population:
        net.generate_empty_genome(4, 1)

    neat.perform_mutation()

    while True:
        bouncy_ball(neat.population, neat.current_gen)

        neat.perform_crossover()
