"""
Collect github projects by programming language
extract trace links between commits and issues
create doc string to source code relationship
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Issue:
    def __init__(self, issue_id: str, desc: str, commit_time):
        self.issue_id = issue_id
        self.desc = "" if pd.isnull(desc) else desc
        self.commit_time = commit_time

    def to_dict(self):
        return {
            "issue_id": self.issue_id,
            "issue_desc": self.desc,
            "commit_time": self.commit_time
        }

    def __str__(self):
        return str(self.to_dict())


class Commit:
    def __init__(self, commit_id, summary, commit_time, file_location):
        self.commit_id = commit_id
        self.summary = summary
        self.commit_time = commit_time
        self.file_location = file_location

    def to_dict(self):
        return {
            "commit_id": self.commit_id,
            "summary": self.summary,
            "commit_time": self.commit_time,
            "file_location": self.file_location
        }

    def __str__(self):
        return str(self.to_dict())
