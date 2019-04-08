import os
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import random as rand
from copy import deepcopy

path, subdir, files = next(os.walk("./img/"))

try:
    input_img = Image.open("./img/" + files[0])
except IOError:
    raise IOError("Input image file was not found.")
else:
    print("Chosen file:", files[0])

INIT_POPULATION = 1500
MIN_POPULATION = 1000

INDIVIDUAL_CAPACITY_OF_GENES = 50

MUTATION_CHANCE = 0.6

NUM_OF_PARENTS = 31
SURVIVORS = 0.8

MIN_RADIUS = 5
MAX_RADIUS = 100
RADIUS_BOUNDS = (MIN_RADIUS, MAX_RADIUS)
COORDINATES_BOUNDS = input_img.size

INPUT_PIXELS = np.array(input_img)

DIFFERENCE_TO_KILL_PARENT = 0.1

MAX_ABS_DIFFERENCE = 765 * COORDINATES_BOUNDS[0] * COORDINATES_BOUNDS[1]


# drawing function
def draw_circle(image: Image, coordinates, radius, color):
    ImageDraw.Draw(image, mode="RGBA").ellipse(
        (
            (coordinates[0] - radius, coordinates[1] - radius),
            (coordinates[0] + radius, coordinates[1] + radius)
        ), color)


class Gene:
    score = None

    def __init__(self):
        self.coordinates = rand.randint(0, COORDINATES_BOUNDS[0]), rand.randint(0, COORDINATES_BOUNDS[1])
        self.radius = rand.randint(RADIUS_BOUNDS[0], RADIUS_BOUNDS[1])
        self.color = rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)

    # mutation
    def mutate(self):
        if MUTATION_CHANCE > rand.random():
            self.coordinates = (rand.randint(0, COORDINATES_BOUNDS[0]), rand.randint(0, COORDINATES_BOUNDS[1]))
        if MUTATION_CHANCE > rand.random():
            self.radius = rand.randint(RADIUS_BOUNDS[0], RADIUS_BOUNDS[1])
        if MUTATION_CHANCE > rand.random():
            self.color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255), rand.randint(100, 255))
        self.gene_fitness()
        return self

    # fitness function for gene
    def gene_fitness(self):
        temp_gene_image = Image.new("RGBA", input_img.size, color=(255, 255, 255, 0))
        draw_circle(temp_gene_image, self.coordinates, self.radius, self.color)
        box_coordinates = (self.coordinates[0] - self.radius, self.coordinates[1] - self.radius,
                           self.coordinates[0] + self.radius, self.coordinates[1] + self.radius)
        self.score = np.sum(np.abs(np.array(ImageChops.difference(temp_gene_image.crop(
            box_coordinates
        ), input_img.crop(box_coordinates
                          ))))) / abs(
            (box_coordinates[2] - box_coordinates[0]) * (box_coordinates[3] - box_coordinates[1]) * 765)
        del temp_gene_image
        del box_coordinates
        return self


class Individual:
    genes = []
    img = None
    score = None

    def __init__(self, random=True):
        if random:
            self.genes = [Gene().gene_fitness() for _ in range(0, INDIVIDUAL_CAPACITY_OF_GENES)]
            self.genes_sort()
        else:
            pass

    # calculating fitness function for Individual
    def update_score(self):
        self.genes_sort()
        self.img = Image.new("RGBA", input_img.size, color=(255, 255, 255, 0))
        for gene in self.genes:
            draw_circle(self.img, gene.coordinates, gene.radius, gene.color)
        self.score = np.sum(np.abs(np.array(ImageChops.difference(input_img, self.img)))) / MAX_ABS_DIFFERENCE
        return self

    # mutation
    def mutate(self):
        for gene in self.genes:
            if MUTATION_CHANCE > rand.random():
                gene.mutate()
        return self

    # sorting genes
    def genes_sort(self):
        self.genes.sort(key=lambda gene: gene.score, reverse=True)
        return self


class Organism:

    def __init__(self):
        self.individuals_array = [Individual().update_score() for _ in range(0, INIT_POPULATION)]
        self.size = INIT_POPULATION
        print("Initiated organism.", "Length: ".rjust(17, " "), str(self.size).rjust(5, " "), sep="")

    # sorting individuals
    def sort(self, sequence_sorted=False):
        if sequence_sorted:
            return self
        self.individuals_array.sort(key=lambda lambda_individual: lambda_individual.score)
        return self

    # selection
    def kill_weak(self):
        before_kill = self.size
        num_of_survivors = int(SURVIVORS * self.size)
        if num_of_survivors > MIN_POPULATION:
            if num_of_survivors < NUM_OF_PARENTS:
                num_of_survivors = NUM_OF_PARENTS
            for i in range(self.size - 1, num_of_survivors - 1, -1):
                del self.individuals_array[i]
                self.size -= 1
            print("[LOG][KILL]".ljust(17, " "), "Length after: ".rjust(19, " "), str(self.size).rjust(5, " "),
                  "█", "died=", str(before_kill - self.size).ljust(7, " "), self.result_fitness(), sep="")
        return self

    # crossing over (reproduction)
    def crossover(self):
        log_born = 0
        for i in range(0, NUM_OF_PARENTS - 1):
            for j in range(i + 1, NUM_OF_PARENTS):
                temp_individual = Individual(random=False)
                for k in range(0, INDIVIDUAL_CAPACITY_OF_GENES):
                    temp_individual.genes.append(
                        min(self.individuals_array[i].genes[k], self.individuals_array[j].genes[k],
                            key=lambda gene_score: gene_score.score))
                temp_individual.genes_sort().update_score()
                self.individuals_array.append(temp_individual)
                self.size += 1
                log_born += 1

        print("[LOG][CROSSOVER]".ljust(17, " "), "Length after: ".rjust(19, " "), str(self.size).rjust(5, " "),
              "█born=", str(log_born).ljust(7, " "), self.result_fitness(), sep="")
        return self

    # mutation
    def mutate(self):
        for i in range(0, self.size):
            current_individual = deepcopy(self.individuals_array[i]).mutate().update_score()
            if self.individuals_array[i].score < current_individual.score:
                del current_individual
            else:
                self.individuals_array[i] = current_individual
        print("[LOG][MUTATE]".ljust(41, " "), "█", " " * 12, self.result_fitness(), sep="")
        return self

    # calculation of fitness function over organism
    def result_fitness(self):
        scores_sum = 0
        for individual in self.individuals_array:
            scores_sum += individual.score
        return scores_sum / self.size

    # saving
    def save(self):
        output_img = Image.new("RGBA", input_img.size, color=(255, 255, 255, 255))
        for individual in self.individuals_array:
            output_img = Image.alpha_composite(output_img, individual.img)
        output_img.save("output.png")

    def selection(self):
        self.sort()
        self.kill_weak()


"""
Algorithm
"""

org = Organism()
generation = 1
org.selection()

while True:
    org.mutate()
    org.sort()
    org.crossover()
    org.selection()
    org.save()
    print("Generation".ljust(13, "-"), str(generation).rjust(5, "-"), "Length:".rjust(17, "-"),
          str(org.size).rjust(6, "-"), "Fitness: ".rjust(12, "▇"), org.result_fitness(),
          sep="")
    generation += 1
