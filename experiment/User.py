"""
@author: JUMP
@date 2021/12/18
@description: 用户画像类，主要用于存储用户的浏览记录
"""


class User:

    def __init__(self, last_week, last_month, last_three_months):
        """
        init

        Parameters
        ----------
        last_week : list
                    近一个月的浏览记录

        last_month : list
                    近一个月的浏览记录

        last_three_months : list
                            近三个月的浏览记录
        """
        self.last_week = last_week
        self.last_month = last_month
        self.last_three_months = last_three_months
