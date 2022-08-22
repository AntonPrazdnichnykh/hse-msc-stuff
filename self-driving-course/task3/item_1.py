from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    _thresh_lower = np.array([23, 100, 100])
    _thresh_upper = np.array([33, 255, 255])
    _thresh_intense = 14000

    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        img, _, _, _ = env.step([0, 0])

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
