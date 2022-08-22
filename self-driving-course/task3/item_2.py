from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    _thresh_lower = np.array([23, 100, 100])
    _thresh_upper = np.array([33, 255, 255])
    _thresh_intense = 10000

    def __init__(self, generated_task):
        super().__init__(generated_task)

    @staticmethod
    def turn(env, side, vel, steps):
        # side: 1 -> left, -1 -> right
        for _ in range(steps):
            env.step([vel, side])
            env.render()

    @staticmethod
    def forward(env, steps=15):
        for _ in range(steps):
            env.step([1, 0])
            env.render()

    def change_line(self, env, side=1, vel=0.3, steps=15):
        self.turn(env, side, vel, steps)
        env.step([1, 0])
        env.render()
        self.turn(env, -side, vel, steps)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        img, _, _, _ = env.step([0, 0])

        # go forward until see utochka
        condition = True
        while condition:
            img, reward, done, info = env.step([1, 0])
            # img in RGB
            # add here some image processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(img, self._thresh_lower, self._thresh_upper)
            if mask.sum() / 255 > self._thresh_intense:
                condition = False

            env.render()

        self.change_line(env, side=1)  # turn to a left line
        self.forward(env, steps=7)  # go forward until pass utochka
        self.change_line(env, side=-1)  # turn back to a right line
        self.forward(env, steps=30)
        env.step([0, 0])
