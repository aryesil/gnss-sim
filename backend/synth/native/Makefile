# backend/synth/native/Makefile
CXX ?= c++
CXXFLAGS := -std=c++20 -O3 -ffast-math -fno-math-errno -Wall -Wextra -fPIC
ifeq ($(OS),Windows_NT)
  CXXFLAGS += -D_USE_MATH_DEFINES
  LIBEXT := dll
  SHARED := -shared -static-libgcc -static-libstdc++ -static
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    LIBEXT := dylib
    SHARED := -dynamiclib
  else ifneq (,$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S))$(findstring UCRT,$(UNAME_S))$(findstring CLANG,$(UNAME_S)))
    CXXFLAGS += -D_USE_MATH_DEFINES
    LIBEXT := dll
    SHARED := -shared -static-libgcc -static-libstdc++ -static
  else
    LIBEXT := so
    SHARED := -shared
  endif
endif
LIB := libgnsssynth.$(LIBEXT)
SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
# -MMD -MP: each object also depends on the headers it includes, so an ABI
# change in abi.h / mixer.hpp rebuilds every user (a header-only edit
# previously left stale objects with the old struct layout).
DEP := $(OBJ:.o=.d)

all: $(LIB)
$(LIB): $(OBJ)
	$(CXX) $(CXXFLAGS) $(SHARED) -o $@ $(OBJ)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<
-include $(DEP)
clean:
	rm -f $(OBJ) $(DEP) $(LIB)
.PHONY: all clean
