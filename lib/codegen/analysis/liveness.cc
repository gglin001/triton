#include <iostream>
#include <climits>
#include <unordered_set>
#include "triton/codegen/instructions.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/instructions.h"
#include "triton/ir/value.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{


void liveness::run(ir::module &mod) {
  intervals_.clear();

  // Assigns index to each instruction
  std::map<ir::value*, slot_index> indices;
  for(ir::function *fn: mod.get_function_list()){
    slot_index index = 0;
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *instr: block->get_inst_list()){
      index += 1;
      indices.insert({instr, index});
    }
  }

  // create live intervals
  for(auto &x: layouts_->get_all()) {
    layout_t* layout = x.second;
    if(layout->type != SHARED)
      continue;
    // users
    std::set<ir::value*> users;
    for(ir::value *v: layout->values){
      users.insert(v);
      for(ir::user *u: v->get_users())
        users.insert(u);
    }
    // compute intervals
    unsigned start = INT32_MAX;
    unsigned end = 0;
    for(ir::value *u: users){
      start = std::min(start, indices.at(u));
      end = std::max(end, indices.at(u));
    }
    intervals_[layout] = segment{start, end};
  }



}

}
}
}
