# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:32:35 2021

@author: Carlos Espa Torres
"""

import logging
from typing import List, Iterator, Union

from segmentation.xy_segmentation import XYSegmentationResults, SegmentationOperation, SegmentationGroup
from utils.utils import check_list_types


class ParsedLevel:
    def __init__(self, parsed_groups: Union[List[str], str, None] = None):
        if check_list_types(parsed_groups, str):
            self.__parsed_groups = parsed_groups
        elif isinstance(parsed_groups, str):
            self.__parsed_groups = [parsed_groups]
        elif parsed_groups is None:
            self.__parsed_groups = list()
        else:
            raise TypeError(''.join(('parsed_groups argument must be of type List[str], str, or NoneType',
                                     f' instead found {parsed_groups} of type {type(parsed_groups)}')))

    @property
    def parsed_groups(self) -> List[str]:
        return self.__parsed_groups

    def add_group(self, new_group: str) -> None:
        if not isinstance(new_group, str):
            raise TypeError(''.join(('new_group arg must be of type str,',
                                     f'instead found {new_group} of type {type(new_group)}')))
        self.__parsed_groups.append(new_group)

    def parse_group(self, segmentation_group: SegmentationGroup, parsed_iterator: Iterator[str]) -> str:
        operation = segmentation_group.segmentation_operation
        if operation == SegmentationOperation.NONE or operation == SegmentationOperation.BEGINNING:
            logging.debug('Operation none or beginning detected')
            parsed_group = next(parsed_iterator)
        elif operation == SegmentationOperation.ROOT_REMOVAL:
            logging.debug('Operation root removal detected')
            parsed_group = ''.join(("{\\", next(parsed_iterator), "{", next(parsed_iterator), "}}"))

        elif operation == SegmentationOperation.X_SEGMENTATION:
            logging.debug('Operation x segmentation detected')
            previous_level = 0
            parsed_group = ''
            for symbol_level in segmentation_group.segmentation_levels:
                if symbol_level == previous_level:
                    parsed_group += next(parsed_iterator)
                elif symbol_level < previous_level:
                    if previous_level > 0:
                        parsed_group = ''.join((parsed_group, "}", next(parsed_iterator)))
                    else:
                        parsed_group += next(parsed_iterator)  # = ''.join((parsed_group, "_{", next(parsed_iterator)))
                else:
                    if previous_level >= 0:
                        parsed_group = ''.join((parsed_group, "^{", next(parsed_iterator)))
                    else:
                        parsed_group += next(parsed_iterator)  # = ''.join((parsed_group, "}", next(parsed_iterator)))
                previous_level = symbol_level
            parsed_group += "}" * ((abs(symbol_level) + symbol_level) // 2)
        elif operation == SegmentationOperation.Y_SEGMENTATION:
            logging.debug('Operation y segmentation detected')
            if len(segmentation_group.segmented_images) == 3:
                numerator, _, denominator = next(parsed_iterator), next(parsed_iterator), next(parsed_iterator)
                parsed_group = ''.join((r"\frac{", numerator, "}{", denominator, "}"))
            elif len(segmentation_group.segmented_images) == 2:
                group1, group2 = next(parsed_iterator), next(parsed_iterator)
                if group2 == 'i':
                    parsed_group = 'i'
                elif group2 == 'j':
                    parsed_group = 'j'
                elif [group1, group2] == ['-', '-']:
                    parsed_group = '='
                else:
                    parsed_group = '='  # group2
        else:
            raise ValueError(f'Operation value ({segmentation_group.segmentation_operation}) out of range')
        logging.debug(f"parsed symbols: {parsed_group}")
        self.add_group(parsed_group)


class XYParser:
    def __init__(self, predicted_array: List[str], xy_segmentation_results: XYSegmentationResults):
        self.__parsed_levels = list()
        self.add_level(ParsedLevel(predicted_array))
        for level_index, segmentation_level in enumerate(reversed(xy_segmentation_results.segmentation_levels)):
            logging.debug(f"Parsing level {level_index}")
            self.add_level(ParsedLevel())
            logging.debug(self.previous_level.parsed_groups)
            expression_iter = iter(self.previous_level.parsed_groups)
            for group_index, segmentation_group in enumerate(segmentation_level.segmentation_groups):
                logging.debug(f"Parsing operation {group_index}")
                self.last_level.parse_group(segmentation_group, expression_iter)

    @property
    def parsed_levels(self) -> List[ParsedLevel]:
        return self.__parsed_levels

    @property
    def last_level(self) -> ParsedLevel:
        return self.parsed_levels[-1]

    @property
    def previous_level(self) -> ParsedLevel:
        """Returns the previous parsed level to the last created level"""
        return self.parsed_levels[-2]

    def add_level(self, new_level: ParsedLevel) -> None:
        if not isinstance(new_level, ParsedLevel):
            raise TypeError('new_level argument must be of type ParsedLevel')
        self.__parsed_levels.append(new_level)


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