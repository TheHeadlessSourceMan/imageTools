#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Tools for converting between different measurement types.
"""
from typing import Union
import re


__UNIT_SPLIT_RE__=re.compile('([0-9.-]*)(.*)')


def convertMeasurements(
    fromValue:Union[float,str],
    toUnits:Union[str,None]=None,
    fromUnits:Union[str,None]=None,
    ppi:int=72
    )->float:
    """
    Convert from one type of graphical measurement units to another

    :param fromValue: can be numeric or text.  If text, it can contain
        input units (set fromUnits=None)
    :param toUnits: units to convert to.  If no units given, assume pixels
    :param fromUnits: units to convert from. If None, derive from fromValue.
        If there are no units given anywhere, assume pixels
    :param ppi: required for converting from screen to actual size
        (make sure this is correct for the display mechanism!)

    Supports:
        'in','inches','inch','"',
        'cm','centimeter','centimeters',
        'mm','millimeters','millimeter',
        'pt','point','points',
        'em','ems',
        'en','ens',
        'px','pixel','pixels',
        'pc','pica','picas','pcs',

    NOTE, key conversions:
        1in = 72pt = 24.5mm = 6pica

    For more info, see:
        https://sizes.com/tools/type.htm
        https://www.w3.org/Style/Examples/007/units.en.html
        https://tex.stackexchange.com/questions/8260/what-are-the-various-units-ex-em-in-pt-bp-dd-pc-expressed-in-mm
    """
    unitFactor={ # convert everything to common factor (inches)
        None:ppi,'':ppi, # default to pixels if nothing specified
        'in':1,'inches':1,'inch':1,'"':1,
        'cm':2.45,'centimeter':2.45,'centimeters':2.45,
        'mm':24.5,'millimeters':24.5,'millimeter':24.5,
        'pt':72,'point':72,'points':72, # TRIVIA: in the days of hand-set type, this was 72.27! # noqa: E501 # pylint: disable=line-too-long
        'em':72,'ems':72,
        'en':144,'ens':144,
        'px':ppi,'pixel':ppi,'pixels':ppi,
        'pc':6,'pica':6,'picas':6,'pcs':6,
        }
    if isinstance(fromValue,str):
        if fromUnits is None:
            m=__UNIT_SPLIT_RE__.match(fromValue)
            fromValue=m.group(1)
            fromUnits=m.group(2)
        fromValue=float(fromValue)
    fromUnits=unitFactor[fromUnits.strip().lower()]
    toUnits=unitFactor[toUnits.strip().lower()]
    return toUnits*fromValue/fromUnits


def cmdline(args):
    """
    Run the command line

    :param args: command line arguments (WITHOUT the filename)
    """
    printHelp=False
    ppi=72
    fromValue=None
    toType='px'
    if not args:
        printHelp=True
    else:
        for arg in args:
            if arg.startswith('-'):
                arg=[a.strip() for a in arg.split('=',1)]
                if arg[0] in ['-h','--help']:
                    printHelp=True
                if arg[0]=='--ppi':
                    ppi=float(arg[1])
                else:
                    print('ERR: unknown argument "'+arg+'"')
            else:
                if fromValue is None:
                    fromValue=arg
                else:
                    toType=arg
        if fromValue is None:
            printHelp=True
        else:
            print(convertMeasurements(fromValue,toType,ppi=ppi),toType)
    if printHelp:
        print('Converts between common measurements')
        print('Usage:')
        print('  measurements.py [options] fromValue toUnits')
        print('Options:')
        print('   --ppi=val ....... set the pixels per inch (default=72)')
        print('Example:')
        print('  measurements.py --ppi=300 6.5in px')


if __name__=='__main__':
    import sys
    cmdline(sys.argv[1:])
