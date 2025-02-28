#!/usr/bin/env python
# -*- coding: utf-8 -*-


import curses
from curses import wrapper
import time
import numpy as np

wall_characters = [
    '█', '▀', '▄', '■', '□', '▪', '▫', '▬', '▮', '▰',
    '▲', '▼', '◄', '►', '◆', '◈', '○', '●', '◐', '◑',
    '◒', '◓', '◢', '◣', '◤', '◥', '★', '▭', '▯', '═',
    '║', '╔', '╗', '╚', '╝', '╦', '╩', '╠', '╣', '╬',
    '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '━',
    '┃', '░', '▒', '▓', '⬚', '❏', '❐', '❑', '❒'
]

snake_head = [
    '◉',  # Fisheye
    '⬤',  # Black circle
    '●',  # Black circle
    '◎',  # Bullseye
    '☻',  # Black smiling face
    '◐',  # Circle with left half black
    '◑',  # Circle with right half black
    '◒',  # Circle with lower half black
    '◓',  # Circle with upper half black
    '▶',  # Black right-pointing triangle
    '◀',  # Black left-pointing triangle
    '▲',  # Black up-pointing triangle
    '▼',  # Black down-pointing triangle
    '◄',  # Black left-pointing pointer
    '►',  # Black right-pointing pointer
    '⚉',  # Circle with dot
    '♦',  # Black diamond
    '❖',  # Black diamond minus white X
    '✪',  # Circled white star
    '☘',  # Shamrock (for fun!)
]

snake_body = [
    '○',  # White circle
    '◌',  # Dotted circle
    '◍',  # Circle with vertical fill
    '◎',  # Bullseye
    '•',  # Bullet
    '∘',  # Ring operator
    '◦',  # White bullet
    '◆',  # Black diamond
    '◇',  # White diamond
    '∙',  # Dot operator
    '▪',  # Black small square
    '□',  # White square
    '◻',  # White medium square
    '⚬',  # Medium white circle
    '◯',  # Large circle
]

apple = [
    '🍎',  # Red apple
    '♥',   # Heart
    '❤',   # Heavy heart
    '●',   # Black circle
    '✦',   # Black four pointed star
    '★',   # Black star
    '✹',   # Heavy teardrop-spoked asterisk
    '✾',   # Heavy teardrop-spoked pinwheel
    '❀',   # Black florette
    '◉',   # Fisheye
    '⭐',   # Star
    '∗',   # Asterisk operator
    '❋',   # Heavy teardrop-spoked asterisk
    '✤',   # Heavy four balloon-spoked asterisk
    '✿',   # Black florette
]


class snake:

    def __init__(self):

        try:
            self.level = int(input("""Choose your game difficulty:
            1. noob
            2. moderate
            3. hardcore

            Your level (1/2/3): """))

            self.environment = int(input(
                                        "Choose your game environment:\n"
                                        "1. Box\n"
                                        "2. Infinity\n"
                                        "Your choice (1/2): "
                                    ))

        except:

            ### defaults
            self.level = 2
            self.environment = 1
            print("Using default parameters")


        ## initialize snake of size 2

        self.snake_head = np.random.choice(snake_head)

        self.snake_body = np.random.choice(snake_body)
        
        self.apple = np.random.choice(apple)
        
        self.wall_size = 0

        self.score = 0
        
        wrapper(self.initialize_game)

    
    def initialize_game(self, stdscr):

        ## initialize curses window 
        #stdscr = curses.initscr()

        ## clear window
        stdscr.clear()

        ## store height and width
        self.height, self.width = stdscr.getmaxyx()

        ## initialize space array
        self.space = np.ones((self.height, self.width))

        ## start game
        
        if self.environment == 1:

            self.build_game_walls()
            

        self.launch(stdscr)
        stdscr.getch()
       

    def launch(self, stdscr):

        ## hide cursor
        curses.curs_set(0)

        ## print box fastest way possible
        if self.environment == 1:
            
            ## using np.argwhere to get all indices where space = 0
            indices = np.argwhere(self.space == 0)
            for i in indices:
                    try:
                        stdscr.addstr(i[0], i[1], self.brick)
                    except curses.error:
                        pass                
            stdscr.refresh()

        
        # Heading direction:
        # +1 => move right, -1 => move left,
        # +2 => move down, -2 => move up.
        self.heading_dir = np.random.choice([-2, -1, 1, 2])

        ## init snake and apple
        self.snake_len  = 3

        ## initialize speed
        self.speed = 1

        # row in [wall_size+3, rows-wall_size-3], col in [wall_size+3, cols-wall_size-3]

        # IMPORTANT: Curses expects coordinates as (y, x). We use height for y and width for x!

        self.snake_loc = np.array([
            np.random.randint(self.wall_size + 3, self.height - self.wall_size - 3),
            np.random.randint(self.wall_size + 3, self.width - self.wall_size - 3)
        ], dtype="int64")

        self.apple_loc = np.array([
            np.random.randint(self.wall_size + 3, self.height - self.wall_size - 3),
            np.random.randint(self.wall_size + 3, self.width - self.wall_size - 3)
        ], dtype="int64")


        # Keep track of snake body coordinates
        self.body_cord = np.zeros((1000, 2), dtype="int64")

        self.body_cord[0, :] = self.snake_loc + self.return_heading()
        
        self.body_cord[1, :] = self.snake_loc + 2 * self.return_heading()
        
        self.play(stdscr)



    def play(self, stdscr):

        while (not self.check_crash()):

            ### listen for key press and update the direction

            key_map = {
                        ord('h'): 'left', ord('l'): 'right',
                        ord('k'): 'up', ord('j'): 'down',
                        curses.KEY_UP: 'up', curses.KEY_DOWN: 'down',
                        curses.KEY_LEFT: 'left', curses.KEY_RIGHT: 'right'
                     }

            key = stdscr.getch()
            action = key_map.get(key, "unknown")

            if action == "up":
                self.heading_dir = 2

            elif action == "down":
                self.heading_dir = -2

            elif action == "left":
                self.heading_dir = -1

            elif action == "right":
                self.heading_dir = 1

            if key == ord('q'):
                break

            ### print curr snake and apple

            ## update snake_loc and body_loc
            ### update position based on heading_dir and speed and if is going out of index let it pass thorough other side

            prev_loc = self.snake_loc 
            self.snake_loc = self.check_bounds(self.snake_loc + self.speed * self.return_heading())

            self.body_cord[1:self.snake_len, :] = self.body_cord[:self.snake_len-1, :]

            self.body_cord[0] = prev_loc

            # Clear screen each frame
            stdscr.clear()

            # Redraw walls if environment == 1
            if self.environment == 1:
                indices = np.argwhere(self.space == 0)
                for (r, c) in indices:
                    try:
                        stdscr.addstr(r, c, self.brick)
                    except curses.error:
                        pass


            ## print head and body

            try:
                stdscr.addstr(self.snake_loc[0], self.snake_loc[1] , self.snake_head)
            except curses.error:
                pass


            for i in range(self.snake_len):

                cordinates = self.check_bounds(self.body_cord[i])
                
                try:
                    stdscr.addstr(cordinates[0], cordinates[1] ,self.snake_body)
                
                except curses.error:
                    pass 

            # Draw apple
            try:
                stdscr.addstr(self.apple_loc[0], self.apple_loc[1], self.apple)
            except curses.error:
                pass

            ### check for apple and head coincide

            if np.array_equal(self.snake_loc, self.apple_loc):

                # Increase length
                self.snake_len += 1
  
                ### reinitialize apple_loc

                self.apple_loc = (np.random.randint(self.wall_size + 3, self.width - self.wall_size - 3), np.random.randint(self.wall_size + 3, self.height - self.wall_size - 3))
                
                ### update speed +1
                self.speed = min(10, 1 + self.score // 3)


            stdscr.refresh()
            time.sleep(0.1)



   #  def check_crash(self):

   #      ### head hits walls
   #      if self.environment == 1:
   #          
   #          if self.snake_loc[0] >=  self.width - self.wall_size or self.snake_loc[0] < self.wall_size:
   #              return True
   #          if self.snake_loc[1] >= self.height - (self.wall_size-1) or self.snake_loc[1] < self.wall_size - 1:
   #              return True


    def check_crash(self):
        # Check if the snake head has hit the walls in environment 1.
        if self.environment == 1:
            # Note: self.snake_loc[0] is the row (y) and must be within [wall_size, height-wall_size)
            if self.snake_loc[0] >= self.height - self.wall_size or self.snake_loc[0] < self.wall_size:
                return True
            # self.snake_loc[1] is the column (x) and must be within [wall_size, width-wall_size)
            if self.snake_loc[1] >= self.width - self.wall_size or self.snake_loc[1] < self.wall_size:
                return True

        
        ### head hits body
        # Head hits its own body
        for i in range(self.snake_len):
            if np.array_equal(self.snake_loc, self.body_cord[i]):
                return True

        return False

    
    def return_heading(self):

        if np.absolute(self.heading_dir) == 2:
            return np.array([0, self.heading_dir // 2])

        else:
            return np.array([self.heading_dir, 0])
    

    def check_bounds(self, coords):
        # If environment is Infinity, wrap around edges.
        if self.environment == 2:
            coords[0] %= self.height
            coords[1] %= self.width
        return coords



    def build_game_walls(self):
        # Choose a random wall character.
        self.brick = np.random.choice(wall_characters)
        self.generate_wall()


    def generate_wall(self):
        # Create wall boundaries with a random thickness.
        wall_size = np.random.randint(2, 5)
        self.wall_size = wall_size  # Save wall thickness for later checks.
        # Set the top and bottom walls.
        self.space[:wall_size, :] = 0
        self.space[-wall_size:, :] = 0
        # Set the left and right walls.
        self.space[:, :wall_size] = 0
        self.space[:, -wall_size:] = 0        






def main():

    # __init__ method runs the game in snake class
    snake()



if __name__ == "__main__":
    main()
