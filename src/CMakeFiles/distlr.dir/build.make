# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /apps/code/dist_lr/dist-lr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /apps/code/dist_lr/dist-lr

# Include any dependencies generated for this target.
include src/CMakeFiles/distlr.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/distlr.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/distlr.dir/flags.make

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o: src/CMakeFiles/distlr.dir/flags.make
src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o: src/OnlinelbfgsMain.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o -c /apps/code/dist_lr/dist-lr/src/OnlinelbfgsMain.cc

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.i"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /apps/code/dist_lr/dist-lr/src/OnlinelbfgsMain.cc > CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.i

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.s"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /apps/code/dist_lr/dist-lr/src/OnlinelbfgsMain.cc -o CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.s

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.requires:

.PHONY : src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.requires

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.provides: src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.requires
	$(MAKE) -f src/CMakeFiles/distlr.dir/build.make src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.provides.build
.PHONY : src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.provides

src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.provides.build: src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o


src/CMakeFiles/distlr.dir/distserver.cc.o: src/CMakeFiles/distlr.dir/flags.make
src/CMakeFiles/distlr.dir/distserver.cc.o: src/distserver.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/distlr.dir/distserver.cc.o"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distlr.dir/distserver.cc.o -c /apps/code/dist_lr/dist-lr/src/distserver.cc

src/CMakeFiles/distlr.dir/distserver.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distlr.dir/distserver.cc.i"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /apps/code/dist_lr/dist-lr/src/distserver.cc > CMakeFiles/distlr.dir/distserver.cc.i

src/CMakeFiles/distlr.dir/distserver.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distlr.dir/distserver.cc.s"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /apps/code/dist_lr/dist-lr/src/distserver.cc -o CMakeFiles/distlr.dir/distserver.cc.s

src/CMakeFiles/distlr.dir/distserver.cc.o.requires:

.PHONY : src/CMakeFiles/distlr.dir/distserver.cc.o.requires

src/CMakeFiles/distlr.dir/distserver.cc.o.provides: src/CMakeFiles/distlr.dir/distserver.cc.o.requires
	$(MAKE) -f src/CMakeFiles/distlr.dir/build.make src/CMakeFiles/distlr.dir/distserver.cc.o.provides.build
.PHONY : src/CMakeFiles/distlr.dir/distserver.cc.o.provides

src/CMakeFiles/distlr.dir/distserver.cc.o.provides.build: src/CMakeFiles/distlr.dir/distserver.cc.o


src/CMakeFiles/distlr.dir/lr.cc.o: src/CMakeFiles/distlr.dir/flags.make
src/CMakeFiles/distlr.dir/lr.cc.o: src/lr.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/distlr.dir/lr.cc.o"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distlr.dir/lr.cc.o -c /apps/code/dist_lr/dist-lr/src/lr.cc

src/CMakeFiles/distlr.dir/lr.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distlr.dir/lr.cc.i"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /apps/code/dist_lr/dist-lr/src/lr.cc > CMakeFiles/distlr.dir/lr.cc.i

src/CMakeFiles/distlr.dir/lr.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distlr.dir/lr.cc.s"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /apps/code/dist_lr/dist-lr/src/lr.cc -o CMakeFiles/distlr.dir/lr.cc.s

src/CMakeFiles/distlr.dir/lr.cc.o.requires:

.PHONY : src/CMakeFiles/distlr.dir/lr.cc.o.requires

src/CMakeFiles/distlr.dir/lr.cc.o.provides: src/CMakeFiles/distlr.dir/lr.cc.o.requires
	$(MAKE) -f src/CMakeFiles/distlr.dir/build.make src/CMakeFiles/distlr.dir/lr.cc.o.provides.build
.PHONY : src/CMakeFiles/distlr.dir/lr.cc.o.provides

src/CMakeFiles/distlr.dir/lr.cc.o.provides.build: src/CMakeFiles/distlr.dir/lr.cc.o


src/CMakeFiles/distlr.dir/util.cc.o: src/CMakeFiles/distlr.dir/flags.make
src/CMakeFiles/distlr.dir/util.cc.o: src/util.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/distlr.dir/util.cc.o"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distlr.dir/util.cc.o -c /apps/code/dist_lr/dist-lr/src/util.cc

src/CMakeFiles/distlr.dir/util.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distlr.dir/util.cc.i"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /apps/code/dist_lr/dist-lr/src/util.cc > CMakeFiles/distlr.dir/util.cc.i

src/CMakeFiles/distlr.dir/util.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distlr.dir/util.cc.s"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /apps/code/dist_lr/dist-lr/src/util.cc -o CMakeFiles/distlr.dir/util.cc.s

src/CMakeFiles/distlr.dir/util.cc.o.requires:

.PHONY : src/CMakeFiles/distlr.dir/util.cc.o.requires

src/CMakeFiles/distlr.dir/util.cc.o.provides: src/CMakeFiles/distlr.dir/util.cc.o.requires
	$(MAKE) -f src/CMakeFiles/distlr.dir/build.make src/CMakeFiles/distlr.dir/util.cc.o.provides.build
.PHONY : src/CMakeFiles/distlr.dir/util.cc.o.provides

src/CMakeFiles/distlr.dir/util.cc.o.provides.build: src/CMakeFiles/distlr.dir/util.cc.o


src/CMakeFiles/distlr.dir/worker.cc.o: src/CMakeFiles/distlr.dir/flags.make
src/CMakeFiles/distlr.dir/worker.cc.o: src/worker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/distlr.dir/worker.cc.o"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/distlr.dir/worker.cc.o -c /apps/code/dist_lr/dist-lr/src/worker.cc

src/CMakeFiles/distlr.dir/worker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/distlr.dir/worker.cc.i"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /apps/code/dist_lr/dist-lr/src/worker.cc > CMakeFiles/distlr.dir/worker.cc.i

src/CMakeFiles/distlr.dir/worker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/distlr.dir/worker.cc.s"
	cd /apps/code/dist_lr/dist-lr/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /apps/code/dist_lr/dist-lr/src/worker.cc -o CMakeFiles/distlr.dir/worker.cc.s

src/CMakeFiles/distlr.dir/worker.cc.o.requires:

.PHONY : src/CMakeFiles/distlr.dir/worker.cc.o.requires

src/CMakeFiles/distlr.dir/worker.cc.o.provides: src/CMakeFiles/distlr.dir/worker.cc.o.requires
	$(MAKE) -f src/CMakeFiles/distlr.dir/build.make src/CMakeFiles/distlr.dir/worker.cc.o.provides.build
.PHONY : src/CMakeFiles/distlr.dir/worker.cc.o.provides

src/CMakeFiles/distlr.dir/worker.cc.o.provides.build: src/CMakeFiles/distlr.dir/worker.cc.o


# Object files for target distlr
distlr_OBJECTS = \
"CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o" \
"CMakeFiles/distlr.dir/distserver.cc.o" \
"CMakeFiles/distlr.dir/lr.cc.o" \
"CMakeFiles/distlr.dir/util.cc.o" \
"CMakeFiles/distlr.dir/worker.cc.o"

# External object files for target distlr
distlr_EXTERNAL_OBJECTS =

bin/distlr: src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o
bin/distlr: src/CMakeFiles/distlr.dir/distserver.cc.o
bin/distlr: src/CMakeFiles/distlr.dir/lr.cc.o
bin/distlr: src/CMakeFiles/distlr.dir/util.cc.o
bin/distlr: src/CMakeFiles/distlr.dir/worker.cc.o
bin/distlr: src/CMakeFiles/distlr.dir/build.make
bin/distlr: /usr/lib/x86_64-linux-gnu/libzmq.so
bin/distlr: /usr/local/lib/libprotobuf.so
bin/distlr: ps-lite/libpslite.a
bin/distlr: /usr/local/lib/libprotobuf.so
bin/distlr: src/CMakeFiles/distlr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/apps/code/dist_lr/dist-lr/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../bin/distlr"
	cd /apps/code/dist_lr/dist-lr/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/distlr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/distlr.dir/build: bin/distlr

.PHONY : src/CMakeFiles/distlr.dir/build

src/CMakeFiles/distlr.dir/requires: src/CMakeFiles/distlr.dir/OnlinelbfgsMain.cc.o.requires
src/CMakeFiles/distlr.dir/requires: src/CMakeFiles/distlr.dir/distserver.cc.o.requires
src/CMakeFiles/distlr.dir/requires: src/CMakeFiles/distlr.dir/lr.cc.o.requires
src/CMakeFiles/distlr.dir/requires: src/CMakeFiles/distlr.dir/util.cc.o.requires
src/CMakeFiles/distlr.dir/requires: src/CMakeFiles/distlr.dir/worker.cc.o.requires

.PHONY : src/CMakeFiles/distlr.dir/requires

src/CMakeFiles/distlr.dir/clean:
	cd /apps/code/dist_lr/dist-lr/src && $(CMAKE_COMMAND) -P CMakeFiles/distlr.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/distlr.dir/clean

src/CMakeFiles/distlr.dir/depend:
	cd /apps/code/dist_lr/dist-lr && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /apps/code/dist_lr/dist-lr /apps/code/dist_lr/dist-lr/src /apps/code/dist_lr/dist-lr /apps/code/dist_lr/dist-lr/src /apps/code/dist_lr/dist-lr/src/CMakeFiles/distlr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/distlr.dir/depend

