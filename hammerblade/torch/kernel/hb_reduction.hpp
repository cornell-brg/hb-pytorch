#ifndef _HB_REDUCTION_H
#define _HB_REDUCTION_H

enum Reduction {
  None,             // Do not reduce
  Mean,             // (Possibly weighted) mean of losses
  Sum,              // Sum losses
  END
};

#endif
