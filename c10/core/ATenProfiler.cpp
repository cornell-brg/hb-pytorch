#include <c10/core/ATenProfiler.h>

#include <map>
#include <vector>

namespace c10 {

ATenProfiler g_aten_profiler;
std::vector<std::string> g_curr_call_stack;

void ATenProfiler::print()
{
  using std::chrono::microseconds;
  using namespace std;

  cout << endl << endl;
  cout << "==========================================================================" << endl;
  cout << " ATen profile results" << endl;
  cout << "==========================================================================" << endl;
  cout << setw(340) << std::left << "Fucntion" << "   " << "Time" << endl;
  double total_time = 0.0;
  for (const auto& p : dict) {
    double ms = p.second.count() / 1000.0;
    auto& stack = p.first;
    // we concat the call stack to get the name
    std::string name;

    // if (stack.size() > 1)
    //   continue; // do not print sub-routine for now...
    if (stack.size() == 1) {
      total_time += ms;
    }

    if (stack.size() > 1) {
      for (size_t i = 1; i < stack.size(); i++) {
        name += "  ";
      }
      name += "|- ";
    }
    name += stack.back();
    cout << setw(340) << std::left << name << "   " << ms / 1000.0 << " s" << endl;
  }
  cout << setw(340) << std::left << "total:" << "   " << total_time / 1000.0 << " s" << endl;
}

void ATenProfiler::add_log(const std::vector<std::string>& stack, std::chrono::microseconds time) {
  if (dict.find(stack) != dict.end()) {
    dict[stack] += time;
  } else {
    dict[stack] = time;
  }
}

ATenProfilerLog::ATenProfilerLog(const std::string& func_name)
  : start(std::chrono::high_resolution_clock::now())
{
  g_curr_call_stack.push_back(func_name);
}

ATenProfilerLog::~ATenProfilerLog()
{
  auto delta = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
  g_aten_profiler.add_log(g_curr_call_stack, delta);
  g_curr_call_stack.pop_back();
}

}
