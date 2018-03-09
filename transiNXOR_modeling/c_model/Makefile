# -*- mode: makefile -*-
# -*- eval: font-lock-fontify-buffer -*-
#
# (C) 2008 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# This work is protected by copyright laws and contains Cadence proprietary
# and confidential information.  No part of this file may be reproduced,
# modified, re-published, used, disclosed or distributed in any way, in any
# medium, whether in whole or in part, without prior written permission from
# Cadence Design Systems, Inc. (Cadence).  If you are a customer of Cadence,
# such permission may be found in a previously executed license and
# maintenance agreement between authorized representatives of you and Cadence.
#
# Notwithstanding any previous agreement, written or otherwise, this file is
# provided to you AS IS with no warranties of any kind, and you agree to bear
# the entire risk of any customizations you make using this file.

ROOT		:= .
BLAS_DIR := ${BLAS_DIR}
LINK_FLAGS=-L$(BLAS_DIR)/lib -Wl,-rpath,$(BLAS_DIR)/lib -lopenblas 

# Include some standard variables.
include $(ROOT)/plugins.variables

# Default rule
default : all 

# Definition of the tfet library
LIB_SRC		:= veriloga_bridge.c device_model.c
LIB_NAME	:= tfet
CFLAGS=-pthread -I$(BLAS_DIR)/include
LIB_LDFLAGS=$(LINK_FLAGS) -pthread
#LIB_LDLIBS=openblas
include $(GNUMAKE_ROOT)/library.rules

# Uncomment the following lines and update the LIB_* variables to 
# create another library.
# LIB_SRC	:= List of C/C++ files [.c, .cc, .cpp, .C]
# LIB_NAME	:= Library name, lib$(LIB_NAME)_sh.so
# LIB_LDFLAGS	:= [OPTIONAL] Flags to be passed to the linker.
# LIB_LDLIBS	:= [OPTIONAL] Extra libraries to be linked into the .so
# include $(GNUMAKE_ROOT)/library.rules

# Include some standard rules
include $(GNUMAKE_ROOT)/std.rules

main: main.c device_model.c
	$(CC) -o $@ $(CFLAGS) $(LINK_FLAGS) -lm -m32 $^
