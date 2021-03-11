import pygame
import numpy as np
import gym
import math
from random import randint
from gym import error, spaces, utils
from gym.spaces import Discrete,Box
from gym.utils import seeding
from gym.envs.classic_control import rendering

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self):
        pygame.init()

        self.play_on = True
        self.window_height = 360
        self.window_width=360
        self.BLOCK_SIZE = 10
        self.BLACK = (0,0,0)
        self.GREEN = (0,255,0)
        self.RED = (255,0,0)

        self.starting_position = 100

        self.fps = 20

        self.snake = []
        self.xSpeed = 10
        self.ySpeed = 0

        self.apple_cords = self.get_new_apple_cord()

        self.surface = pygame.display.set_mode((self.window_width, self.window_height))
        self.discrete_actions = [0,1,2,3]

        self.action_space = Discrete(4)
        self.action = np.random.choice(self.discrete_actions)
        self.observation_space=Box(low=0,high=255,shape=(self.window_height,self.window_width),
        dtype=np.uint8)
        self.n_steps = 0
        self.max_steps=200
        self.viewer = None

    def show_score(self):
        font = pygame.font.Font('freesansbold.ttf', 12)
        text = font.render('Score: '+str(len(self.snake)-1),True,(255,255,255))
        return text

    def draw_apple(self,appleX,appleY):
        ##draw a red rect to represent an apple
        pygame.draw.rect(self.surface,self.RED,pygame.Rect(appleX,appleY,10,10))

    def death(self):
        head_x = self.snake[len(self.snake)-1][0]
        head_y = self.snake[len(self.snake)-1][1]

        ## if head is outside of screen, then die
        if head_x > self.window_width - self.BLOCK_SIZE or head_x < 0 or head_y > self.window_height -self.BLOCK_SIZE or head_y < 0:
            print("death")
            return True
        
        ## if head touches body of snake, then die
        for x_y in self.snake[0:len(self.snake)-1]:
            #x_y is tuple of the x and y coordinates
            if x_y == self.snake[len(self.snake)-1]:
                print("death")
                return True
    
        return False

    def get_new_apple_cord(self):
        not_in_snake = False

        while not not_in_snake:
            appleX = math.ceil(randint(0,self.window_width-10)/10.0) *10
            appleY = math.ceil(randint(0,self.window_height-10)/10.0) *10
            
            apple_cords = (appleX,appleY)

            if apple_cords not in self.snake:
                not_in_snake = True

        return apple_cords 

    ##draw game to surface
    def draw_surface(self,appleX,appleY):
        self.surface.fill(self.BLACK)
        self.draw_snake()
        self.draw_apple(appleX,appleY)
        text = self.show_score()
        self.surface.blit(text,(0,0))
        pygame.display.update()

    def draw_snake(self):
        for i in range(0,len(self.snake)):
            pygame.draw.rect(self.surface, self.GREEN, pygame.Rect(self.snake[i][0], self.snake[i][1], 10, 10)) 

    def step(self, action):
        self.n_steps += 1
        self.play_on = False

        eaten = False

        # if player moves left 
        if action == 0:
            self.xSpeed = 10
            self.ySpeed = 0
        elif action == 1:
            self.xSpeed = -10
            self.ySpeed = 10
        elif action == 2:
            self.ySpeed = 10
            self.xSpeed = 0
        elif action == 3:
            self.ySpeed = -10
            self.xSpeed = 0

        xValue = self.snake[len(self.snake)-1][0] + self.xSpeed
        yValue = self.snake[len(self.snake)-1][1] + self.ySpeed

        self.snake.append((xValue,yValue))

        if xValue == self.apple_cords[0] and yValue == self.apple_cords[1]:
            eaten = True
            self.apple_cords = self.get_new_apple_cord()
        
        if eaten == False:
            self.snake.pop(0)

        self.draw_surface(self.apple_cords[0],self.apple_cords[1])

        observation = self.get_state()
        info = {}

        if self.death():
            reward = -2.0
            done = True
            self.reset()
        elif self.n_steps > self.max_steps:
            done = True
            reward = -1.0
            self.reset()
        elif eaten == True:
            done = False
            reward = +2.0
        else:
            reward = 0
            done= False
        return observation, reward, done, info

    
    def get_state(self):
        state = np.fliplr(np.flip(np.rot90(pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8))))
        return state

    def reset(self):
        self.snake = []
        self.snake.append((100,100))
        self.apple_cords = self.get_new_apple_cord()
        self.action = -1
        self.n_step = 0
        self.draw_surface(self.apple_cords[0],self.apple_cords[1])
  
    def render(self, mode='human', close=False):
        img = self.get_state()
        if mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        elif mode == 'rgb_array':
            return img