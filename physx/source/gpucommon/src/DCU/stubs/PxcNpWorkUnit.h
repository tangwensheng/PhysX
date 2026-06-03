// HIP stub: redirect to the real PxcNpWorkUnit.h
// Lookup with relative path from -I$PHYSX_ROOT/source/lowlevel/common/include
#pragma once
#include "pipeline/PxcNpWorkUnit.h"

// Additional constants needed by GPU kernel files
#define DY_SC_FLAG_SPRING               0x0001
#define DY_SC_FLAG_ACCELERATION_SPRING  0x0002
#define DY_SC_FLAG_OUTPUT_FORCE         0x0004
#define DY_SC_FLAG_KEEP_BIAS            0x0008
#define DY_SC_FLAG_INEQUALITY           0x0010
#define DY_SC_FLAG_ORTHO_TARGET         0x0020
#define DY_SC_FLAG_ROT_EQ               0x0040
