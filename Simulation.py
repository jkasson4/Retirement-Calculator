# ------------------------------------------------------
# Simulation.py
# For use with Retirement Calculator
# Author: John Kasson
# ------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math

class Simulation:
    def __init__(self, income, savings, personal, market):
        self.income = income
        self.savings = savings
        self.personal = personal
        self.market = market

        self.savings_percentage = np.array([self.savings.get_savings_contributions_no_match(0)] * self.personal.get_num_steps())
        self.contributions = np.array([self.savings.get_savings_contributions(0) * (self.income.get_post_tax_income() / 26)] * self.personal.get_num_steps())
        self.contributions_no_match = np.array([self.savings.get_savings_contributions_no_match(0) * (self.income.get_post_tax_income() / 26)] * self.personal.get_num_steps())
        self.net_worth = np.array([self.savings.get_starting_balance()] * self.personal.get_num_steps())
        self.net_worth_no_interest = np.array([self.savings.get_starting_balance()] * self.personal.get_num_steps())
        self.net_worth_no_expenses = np.array([self.savings.get_starting_balance()] * self.personal.get_num_steps())
        self.market_evolution = np.array([self.market.get_next_evolution()] * self.personal.get_num_steps())
        self.expenses = np.array([[0.0] * (self.savings.get_num_accounts() + 1)] * self.personal.get_num_steps())
    
    def run(self):
        
        # Run market simulation
        steps = self.personal.get_num_steps()
        
        # Calculate biweekly contributions
        for n in range(1, steps):
            
            # Calculate post-tax income
            if(n % 26 == 0):
                self.income.pay_raise()
            post_tax_income = self.income.get_post_tax_income()
            
            # Determine contributions and savings percentage
            self.savings_percentage[n] = self.savings.get_savings_contributions_no_match(n)
            self.contributions[n] = self.savings.get_savings_contributions(n) * (post_tax_income / 26)
            self.contributions_no_match[n] = self.savings.get_savings_contributions_no_match(n) * (post_tax_income / 26)

            # Determine market evolution
            self.market_evolution[n] = self.market.get_next_evolution()

            # Determine expenses
            savings_expenses = self.savings.get_expenses(n, self.net_worth[n-1])
            personal_expenses = self.personal.get_expenses(n)
            expenses = np.append(savings_expenses, personal_expenses)
            self.expenses[n] = self.expenses[n-1] + expenses

            # Divide personal expenses into savings accounts
            expenses_for_calculation = savings_expenses + self.savings.divide_expenses(personal_expenses)
            
            # Determine net worth - Update with market
            self.net_worth[n] = self.net_worth[n-1] * (1 + self.market_evolution[n] / 26) + self.contributions[n] - expenses_for_calculation
            self.net_worth_no_interest[n] = self.net_worth_no_interest[n-1] + self.contributions[n]
            self.net_worth_no_expenses[n] = self.net_worth_no_expenses[n-1] * (1 + self.market_evolution[n] / 26) + self.contributions[n]
    
    def output(self):

        # Define order of colors
        colors = ['b', 'r', 'g']
        
        # Reformat from biweekly to yearly timespan
        years_range = np.array(list(map(lambda x:x / 26, range(self.personal.current_age * 26, self.personal.current_age * 26 + self.personal.get_num_steps()))))

        # Total net worth by step
        savings_total = np.array(list(map(lambda x:sum(x), self.net_worth)))
        savings_total_no_interest = np.array(list(map(lambda x:sum(x), self.net_worth_no_interest)))
        savings_total_no_expenses = np.array(list(map(lambda x:sum(x), self.net_worth_no_expenses)))

        # Restructure catagorical net worth
        savings_by_type  = np.array([[0.0] * self.personal.get_num_steps()] * self.savings.get_num_accounts())
        for i in range(len(self.net_worth[0])):
            for j in range(self.personal.get_num_steps()):
                savings_by_type[i][j] = self.net_worth[j][i]
        
        # Plot total net worth
        plt.plot(years_range, savings_total, colors[0], label='Total Net Worth', zorder=2)
        plt.plot(years_range, savings_total_no_interest, colors[1], label='Net Worth From Contributions', zorder=1)
        plt.plot(years_range, savings_total_no_expenses, colors[2], label='Net Worth Without Expenses', zorder=0)
        plt.ticklabel_format(axis='y', style='plain')
        plt.title("Net Worth by " + str(self.personal.retirement_age))
        plt.xlabel('Age')
        plt.ylabel('Net Worth')
        plt.legend(loc='upper left')
        plt.show()

        # Plot net worth by savings type
        plt.plot(years_range, savings_total, colors[0], label='Total Net Worth', zorder=len(savings_by_type)+1)
        for i in range(len(savings_by_type)):
            plt.plot(years_range, savings_by_type[i], colors[i+1], label=self.savings.get_names()[i], zorder=len(savings_by_type)-i)
        plt.ticklabel_format(axis='y', style='plain')
        plt.title("Net Worth By Account Type")
        plt.xlabel('Age')
        plt.ylabel('Net Worth')
        plt.legend(loc='upper left')
        plt.show()

        # Combine biweekly contributions
        biweekly_contributions = np.array(list(map(lambda x:sum(x), self.contributions)))
        biweekly_contributions_no_match = np.array(list(map(lambda x:sum(x), self.contributions_no_match)))
        
        # Plot biweekly contribution
        plt.plot(years_range, biweekly_contributions, colors[0], label='With Employer Match', zorder=1)
        plt.plot(years_range, biweekly_contributions_no_match, colors[1], label='Without Employer Match', zorder=0)
        plt.ticklabel_format(axis='y', style='plain')
        plt.title("Biweekly Contribution")
        plt.xlabel('Age')
        plt.ylabel('Contribution')
        plt.legend(loc='upper left')
        plt.show()

        # Total savings percentages by step
        percentage_savings_total = np.array(list(map(lambda x:sum(x), self.savings_percentage)))

        # Restructure savings percentages
        percentage_savings_by_type = np.array([[0.0] * self.personal.get_num_steps()] * self.savings.get_num_accounts())
        for i in range(len(self.savings_percentage[0])):
            for j in range(self.personal.get_num_steps()):
                percentage_savings_by_type[i][j] = self.savings_percentage[j][i]

        # Plot percentage of income to savings
        plt.plot(years_range, percentage_savings_total, 'b', label='All Savings', zorder=len(savings_by_type)+1)
        for i in range(len(percentage_savings_by_type)):
            plt.plot(years_range, percentage_savings_by_type[i], colors[i+1], label=self.savings.get_names()[i], zorder=len(percentage_savings_by_type)-i)
        plt.ticklabel_format(axis='y', style='plain')
        plt.title("Percentage of Income to Savings")
        plt.xlabel('Age')
        plt.ylabel('Percentage of Income')
        plt.legend(loc='upper left')
        plt.show()

        # Plot market growth
        plt.plot(years_range, self.market_evolution, 'b')
        plt.ticklabel_format(axis='y', style='plain')
        plt.title("Market Growth Rate")
        plt.xlabel('Age')
        plt.ylabel('Growth Rate')
        plt.show()

    def get_retirement_savings(self):
        return sum(self.net_worth[self.personal.get_num_steps() - 1])

    def get_withdrawal_safe_current(self):
        return self.savings.get_withdrawal(self.net_worth[self.personal.get_num_steps() - 1] * (0.04 / 26), self.personal.retirement_age, self.personal.marriage_status)
        
    def get_withdrawal_safe_over_60(self):
        return self.savings.get_withdrawal(self.net_worth[self.personal.get_num_steps() - 1] * (0.04 / 26), 60, self.personal.marriage_status)

class Income:
    def __init__(self, pre_tax_income, annual_raise, fed_tax_bracket, fed_tax_rate):
        self.pre_tax_income = pre_tax_income
        self.annual_raise = annual_raise + np.random.normal(0, 0.01)
        self.fed_tax_bracket = fed_tax_bracket
        self.fed_tax_rate = fed_tax_rate

        # Adjust bracket for calculation
        for i in range(len(self.fed_tax_bracket) - 1, 0, -1):
            self.fed_tax_bracket[i] -= self.fed_tax_bracket[i-1]

    def get_post_tax_income(self):
        return self.helper_post_tax_income(self.pre_tax_income, 0)
        
    def helper_post_tax_income(self, income, i):
        if(i > len(self.fed_tax_bracket) - 1 or income < self.fed_tax_bracket[i]):
            return (1 - self.fed_tax_rate[i]) * income
        return (1 - self.fed_tax_rate[i]) * self.fed_tax_bracket[i] + self.helper_post_tax_income(income - self.fed_tax_bracket[i], i + 1)
        
    def pay_raise(self):
        self.pre_tax_income = self.pre_tax_income * (1 + self.annual_raise + np.random.normal(0, 0.01))

class Savings:
    def __init__(self, roth401k, brokerage):
        self.roth401k = roth401k
        self.brokerage = brokerage

    def get_savings_contributions(self, n):
        return np.array([self.roth401k.get_savings_contributions(n), self.brokerage.get_savings_contributions(n)])

    def get_savings_contributions_no_match(self, n):
        return np.array([self.roth401k.get_savings_contributions_no_match(n), self.brokerage.get_savings_contributions(n)])

    def get_withdrawal(self, savings, age, marriage_status):
        if(age < 59.5):
            return self.brokerage.get_withdrawal_amount(savings[1] + 0.5 * savings[0], marriage_status)
        return self.roth401k.get_withdrawal_amount(savings[0], age) + self.brokerage.get_withdrawal_amount(savings[1], marriage_status)

    def get_expenses(self, n, holdings):
        return np.array([0, self.brokerage.get_expenses(n, holdings[1])])

    # Always splits non-retirement expenses among non-roth accounts.
    # Optimal as gained income is not taxed in roth accounts.
    def divide_expenses(self, amount):
        return np.array([0, amount])

    def get_starting_balance(self):
        return np.array([self.roth401k.starting_balance, self.brokerage.starting_balance])

    def get_names(self):
        return np.array([self.roth401k.name, self.brokerage.name])

    def get_num_accounts(self):
        return 2

class Roth401K:
    def __init__(self, starting_balance, contribution, growth_rate, emp_match_bracket, emp_match_rate):
        self.starting_balance = starting_balance
        self.contribution = contribution
        self.growth_rate = growth_rate
        self.emp_match_bracket = emp_match_bracket
        self.emp_match_rate = emp_match_rate
        self.name = "Roth 401k"

        # Adjust bracket for calculation
        for i in range(len(self.emp_match_bracket) - 1, 0, -1):
            self.emp_match_bracket[i] -= self.emp_match_bracket[i-1]

    # Returns percentage contribution to roth 401k after growth rate and employer match
    def get_savings_contributions(self, n):
        return self.helper_savings_contributions(round(self.contribution * (1 + self.growth_rate) ** int(n / 26), 2), 0)
    
    def helper_savings_contributions(self, percentage_contribution, i):
        if(i > len(self.emp_match_bracket) - 1 or percentage_contribution < self.emp_match_bracket[i]):
            return (1 + self.emp_match_rate[i]) * percentage_contribution
        return (1 + self.emp_match_rate[i]) * self.emp_match_bracket[i] + self.helper_savings_contributions(percentage_contribution - self.emp_match_bracket[i], i + 1)

    def get_savings_contributions_no_match(self, n):
        return round(self.contribution * (1 + self.growth_rate) ** int(n / 26), 2)
    
    # Calculates withdrawal amount after unqualified withdrawal fee
    def get_withdrawal_amount(self, amount, age):
        if(age < 59.5):
            return amount * 0.9
        return amount

class Brokerage:
    def __init__(self, starting_balance, contribution, growth_rate, expense_ratio, gains_bracket_single, gains_bracket_married, gains_rate):
        self.starting_balance = starting_balance
        self.contribution = contribution
        self.growth_rate = growth_rate
        self.expense_ratio = expense_ratio
        self.gains_bracket_single = gains_bracket_single
        self.gains_bracket_married = gains_bracket_married
        self.gains_rate = gains_rate
        self.name = "Brokerage"

        # Adjust brackets for calculation
        for i in range(len(self.gains_bracket_single) - 1, 0, -1):
            self.gains_bracket_single[i] -= self.gains_bracket_single[i-1]
            self.gains_bracket_married[i] -= self.gains_bracket_married[i-1]

    # Returns percentage contribution to brokerage after growth rate
    def get_savings_contributions(self, n):
        return round(self.contribution * (1 + self.growth_rate) ** int(n / 26), 2)
        
    # Calculates the withdrawal amount after capital gains tax.
    # Assumes all stocks are held for at least 1 year prior to withdrawal.
    def get_withdrawal_amount(self, amount, marriage_status):
        if(marriage_status == False):
            return self.helper_withdrawal_amount_single(amount, 0)
        return self.helper_withdrawal_amount_married(amount, 0)

    def helper_withdrawal_amount_single(self, amount, i):
        if(i > len(self.gains_bracket_single) - 1 or income < self.gains_bracket_single[i]):
            return (1 - self.gains_rate[i]) * amount
        return (1 - self.gains_rate[i]) * self.gains_bracket_single[i] + self.helper_withdrawal_amount_single(amount - self.gains_bracket_single[i], i + 1)

    def helper_withdrawal_amount_married(self, amount, i):
        if(i > len(self.gains_bracket_married) - 1 or amount < self.gains_bracket_married[i]):
            return (1 - self.gains_rate[i]) * amount
        return (1 - self.gains_rate[i]) * self.gains_bracket_married[i] + self.helper_withdrawal_amount_married(amount - self.gains_bracket_married[i], i + 1)

    def get_expenses(self, n, holdings):
        if(n%26 == 0):
            return holdings * self.expense_ratio
        return 0

 
class Personal:
    def __init__(self, current_age, retirement_age, marriage_status, expenses):
        self.current_age = current_age
        self.retirement_age = retirement_age
        self.marriage_status = marriage_status
        self.expenses = list(map(lambda x,:[(x[0] - self.current_age) * 26, x[1]], expenses))

        # Restructure expense age for calculation

    # Returns the number of steps in the simulation
    def get_num_steps(self):
        return (self.retirement_age - self.current_age) * 26

    def get_expenses(self, n):
        total_expenses = 0
        for i in range(len(self.expenses)):
            if(n == self.expenses[i][0]):
                total_expenses += self.expenses[i][1]
        return total_expenses

class Market:
    def __init__(self, growth_rate, growth_uncertainty, cycle_time, cycle_uncertainty, volatility, volatility_uncertainty, fluctuation_strength):
        self.growth_rate = growth_rate + np.random.normal(0, growth_uncertainty)
        self.cycle_speed = cycle_time / 13 * math.pi * np.random.normal(0, 0.25)
        self.cycle_position = np.random.rand() * 26
        self.cycle_uncertainty = cycle_uncertainty
        self.volatility = volatility * np.random.normal(0, volatility_uncertainty)
        self.fluctuation_strength = fluctuation_strength

    # Progresses market cycle when called.
    # Returns the growth rate at the current market step.
    def get_next_evolution(self):
        self.cycle_position = self.cycle_position + np.random.normal(self.cycle_speed, self.cycle_uncertainty)
        fluctuation = np.random.normal(0, self.fluctuation_strength)
        return self.growth_rate + self.volatility * math.sin(self.cycle_position + self.cycle_speed) + fluctuation