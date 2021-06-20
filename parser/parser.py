# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:32:35 2021

@author: Carlos Espa Torres
"""

import logging


def parse(equation_structure):
    """Takes a prediction of symbols and parses it to LaTeX"""
    latex_string = r""
    # take array of symbols
    # FIXME: fix upstream grouping
    construction_array = [equation_structure[-1][1]]
    for level_index, (operations, images) in enumerate(reversed(equation_structure)):
        logging.debug(f"Parsing level {level_index}")
        logging.debug([operations, construction_array[-1]])
        expression_iter = iter(construction_array[-1])
        construction_array.append([])
        for operation_index, operation in enumerate(operations):
            logging.debug(f"Parsing operation {operation_index}")
            if operation[0] == 'n':
                construction_array[-1].append(next(expression_iter))
                logging.debug(f"parsed symbols: {construction_array[-1][-1]}")
            elif operation[0] == 'r':
                symbol_array = next(expression_iter)
                construction_array[-1].append("{\\" + symbol_array[0] + "{" + symbol_array[-1] + "}}")
                logging.debug(f"parsed symbols: {construction_array[-1][-1]}")
            elif operation[0] == 'x':
                previous_level = 0
                expression = r"{"
                symbol_iter = iter(next(expression_iter))
                for symbol_index, symbol_level in enumerate(operation[1]):
                    if symbol_level == previous_level:
                        expression += next(symbol_iter)
                    elif symbol_level < previous_level:
                        if previous_level > 0:
                            expression += "{" + next(symbol_iter)
                        else:
                            expression += "}_{" + next(symbol_iter)
                    else:
                        if previous_level >= 0:
                            expression += "}^{" + next(symbol_iter)
                        else:
                            expression += "}" + next(symbol_iter)
                expression += "}" * (abs(symbol_level) + 1)
                construction_array[-1].append([expression])
                logging.debug(f"parsed symbols: {construction_array[-1][-1]}")
            elif operation[0] == 'y':
                # TODO: find a way to know number of levels, maybe add them again
                symbol_array = next(expression_iter)
                logging.debug(symbol_array)
                if len(symbol_array) == 3:
                    expression = r"{\frac{" + symbol_array[0] + "}{" + symbol_array[-1] + "}}"
                if len(symbol_array) == 2:
                    if symbol_array == ['-', '-']:
                        expression = '='
                    else:
                        expression = symbol_array[-1]
                construction_array[-1].append([expression])
                logging.debug(f"parsed symbols: {construction_array[-1][-1]}")
            elif operation[0] == 'b':
                latex_string = construction_array[-2][-1]

    logging.debug(f"Parsed expression: {latex_string}")
    return latex_string
