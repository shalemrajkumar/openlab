#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import curses
from curses import wrapper
import time
import random
from collections import deque

# Constants for better maintainability
WALL_CHARS = [
    '█', '▀', '▄', '■', '□', '▪', '▫', '▬', '▮', '▰',
    '▲', '▼', '◄', '►', '◆', '◈', '○', '●', '◐', '◑',
    '◒', '◓', '◢', '◣', '◤', '◥', '★', '▭', '▯', '═',
    '║', '╔', '╗', '╚', '╝', '╦', '╩', '╠', '╣', '╬'
]

SNAKE_HEADS = ['◉', '⬤', '●', '◎', '☻', '◐', '◑', '◒', '◓']
SNAKE_BODIES = ['○', '◌', '◍', '◎', '•', '∘', '◦', '◆', '◇']
APPLES = ['🍎', '♥', '❤', '●', '✦', '★', '✹', '✾', '❀']

class SnakeGame:
    def __init__(self):
        self._get_user_settings()
        self._init_game_params()
        wrapper(self._initialize_game)

    def _get_user_settings(self):
        """Validate and get user input for game settings"""
        self.level = self._get_valid_input(
            "Choose difficulty (1-3):\n1. Easy\n2. Medium\n3. Hard\n> ",
            [1, 2, 3],
            default=2
        )
        
        self.environment = self._get_valid_input(
            "Choose environment:\n1. Box\n2. Infinite\n> ",
            [1, 2],
            default=1
        )

    def _get_valid_input(self, prompt, valid_choices, default):
        """Helper for validated user input"""
        while True:
            try:
                value = int(input(prompt))
                if value in valid_choices:
                    return value
                print(f"Invalid input. Please enter {valid_choices}")
            except (ValueError, EOFError):
                print(f"Using default: {default}")
                return default

    def _init_game_params(self):
        """Initialize game parameters"""
        self.snake_head = random.choice(SNAKE_HEADS)
        self.snake_body = random.choice(SNAKE_BODIES)
        self.apple_char = random.choice(APPLES)
        self.wall_thickness = 0
        self.score = 0
        self.direction = random.choice([-2, -1, 1, 2])
        self.snake_length = 3
        self.delay = 0.15 - (self.level * 0.03)  # Adjust speed based on level
        self.body = deque(maxlen=100)  # Efficient FIFO structure

    def _initialize_game(self, stdscr):
        """Setup curses environment"""
        stdscr.clear()
        curses.curs_set(0)
        stdscr.nodelay(1)
        
        self.height, self.width = stdscr.getmaxyx()
        self._generate_walls()
        
        # Initial positions
        y = random.randint(self.wall_thickness, self.height - self.wall_thickness)
        x = random.randint(self.wall_thickness, self.width - self.wall_thickness)
        self.head = np.array([y, x])
        self._place_apple()
        
        self._game_loop(stdscr)

    def _generate_walls(self):
        """Generate walls for box environment"""
        if self.environment == 1:
            self.wall_thickness = random.randint(1, 3)
            self.wall_char = random.choice(WALL_CHARS)

    def _place_apple(self):
        """Place apple in valid position"""
        while True:
            y = random.randint(self.wall_thickness, self.height - self.wall_thickness)
            x = random.randint(self.wall_thickness, self.width - self.wall_thickness)
            self.apple_pos = np.array([y, x])
            # Ensure apple doesn't spawn on snake
            if not np.array_equal(self.apple_pos, self.head) and not any(np.array_equal(self.apple_pos, seg) for seg in self.body):
                break

    def _game_loop(self, stdscr):
        """Main game loop"""
        while True:
            self._process_input(stdscr)
            if self._move() or self._check_collision():
                break
            self._draw_frame(stdscr)
            time.sleep(self.delay)

    def _process_input(self, stdscr):
        """Handle keyboard input"""
        key = stdscr.getch()
        directions = {
            curses.KEY_UP: 2, curses.KEY_DOWN: -2,
            curses.KEY_LEFT: -1, curses.KEY_RIGHT: 1,
            ord('w'): 2, ord('s'): -2, ord('a'): -1, ord('d'): 1
        }
        if key in directions and directions[key] != -self.direction:
            self.direction = directions[key]

    def _move(self):
        """Update snake position"""
        self.body.appendleft(self.head.copy())
        delta = [0, 0]
        if abs(self.direction) == 2:  # Vertical movement
            delta[0] = self.direction // 2
        else:  # Horizontal movement
            delta[1] = self.direction
        
        self.head += delta
        self._handle_boundaries()
        
        # Check apple collision
        if np.array_equal(self.head, self.apple_pos):
            self.snake_length += 1
            self.score += 1
            self.delay = max(self.delay * 0.95, 0.03)  # Gradually increase speed
            self._place_apple()
            return False
        return False

    def _handle_boundaries(self):
        """Handle screen boundaries based on environment"""
        if self.environment == 1:  # Solid walls
            self.head[0] = max(min(self.head[0], self.height - self.wall_thickness - 1)
            self.head[1] = max(min(self.head[1], self.width - self.wall_thickness - 1)
        else:  # Wrapping boundaries
            self.head %= [self.height, self.width]

    def _check_collision(self):
        """Check for collisions"""
        # Wall collision
        if self.environment == 1 and (
            self.head[0] <= self.wall_thickness or
            self.head[0] >= self.height - self.wall_thickness or
            self.head[1] <= self.wall_thickness or
            self.head[1] >= self.width - self.wall_thickness
        ):
            return True
        
        # Self collision
        return any(np.array_equal(self.head, segment) for segment in self.body)

    def _draw_frame(self, stdscr):
        """Render game state"""
        stdscr.clear()
        
        # Draw walls
        if self.environment == 1:
            for y in range(self.wall_thickness):
                stdscr.addstr(y, 0, self.wall_char * self.width)
            for y in range(self.height - self.wall_thickness, self.height):
                stdscr.addstr(y, 0, self.wall_char * self.width)
            for y in range(self.height):
                stdscr.addstr(y, 0, self.wall_char * self.wall_thickness)
                stdscr.addstr(y, self.width - self.wall_thickness, self.wall_char * self.wall_thickness)
        
        # Draw snake
        try:
            stdscr.addstr(self.head[0], self.head[1], self.snake_head)
            for segment in self.body:
                stdscr.addstr(segment[0], segment[1], self.snake_body)
            stdscr.addstr(self.apple_pos[0], self.apple_pos[1], self.apple_char)
        except curses.error:
            pass
        
        stdscr.refresh()

def main():
    SnakeGame()

if __name__ == "__main__":
    main()
