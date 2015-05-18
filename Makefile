EXECUTABLES = MCMC metropolis MCMC2
CC=gcc49

all: $(EXECUTABLES)

LDFLAGS += $(foreach librarydir,$(subst :, ,$(LD_LIBRARY_PATH)),-L$(librarydir))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  LDFLAGS += -lrt -lOpenCL -lm
  CFLAGS += -Wall -std=gnu99 -g -O2
endif
ifeq ($(UNAME_S),Darwin)
  LDFLAGS +=  -framework OpenCL -lm
  CFLAGS += -Wall -std=c99 -g -O2
endif

ifdef OPENCL_INC
  CPPFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  LDFLAGS = -L$(OPENCL_LIB)
endif

MCMC.o: MCMC.c cl-helper.h timing.h
MCMC2.o: MCMC2.c cl-helper.h timing.h 
cl-helper.o: cl-helper.c cl-helper.h
metropolis.o: metropolis.c timing.h
rng.o: rng.c
plotInfoPrintOut.o: plotInfoPrintOut.c
dataAnalysis.o: dataAnalysis.c
printTimeSeries.o: printTimeSeries.c

MCMC: MCMC.o cl-helper.o plotInfoPrintOut.o dataAnalysis.o printTimeSeries.o

MCMC2: MCMC2.o cl-helper.o plotInfoPrintOut.o dataAnalysis.o printTimeSeries.o

metropolis: metropolis.o rng.o plotInfoPrintOut.o dataAnalysis.o printTimeSeries.o

clean:
	rm -f $(EXECUTABLES) *.o
