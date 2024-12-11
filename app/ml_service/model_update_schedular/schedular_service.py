from typing import Callable
from apscheduler.schedulers.background import BackgroundScheduler


class SchedulerService:
    """Manages dynamic scheduling of tasks."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def add_task(self, name: str, func: Callable, schedule: dict):
        """
        Schedule a task dynamically.
        Args:
            name (str): Unique task name.
            func (Callable): Task function to execute.
            schedule (dict): Cron-style schedule (e.g., {'hour': 0, 'minute': 0}).
        """
        self.scheduler.add_job(func, 'cron', id=name, **schedule)
        print(f"Task '{name}' scheduled with cron: {schedule}.")

    def remove_task(self, name: str):
        """Remove a scheduled task by name."""
        self.scheduler.remove_job(name)
        print(f"Task '{name}' removed.")

    def shutdown(self):
        """Shut down the scheduler."""
        self.scheduler.shutdown()
        print("Scheduler shut down.")
