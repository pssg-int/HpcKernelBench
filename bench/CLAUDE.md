
## NERSC Filesystem Safety (REQUIRED — admin warning received 2026-08-07)

Never recursively traverse `/`, `/global`, `/global/cfs`, `/global/homes`,
`/pscratch`, `/opt`, `/usr`, `/cvmfs`, or any other shared top-level directory.
This covers: `find`, `bfs`, `fd`, `tree`, recursive `du`, `rg --files`,
recursive `grep`, recursive `ls`, globstar expansion, and recursive traversal
in Python or any other language.

Before searching, identify a bounded root inside the current workspace or a
known project or data directory. Constrain depth and filename patterns.

To locate software use `command -v`, `type -a`, `module spider`, package
metadata, or known environment prefixes. Do not search mounted filesystems for
executables or libraries. A compute allocation is not permission for an
unbounded traversal of a shared filesystem.
