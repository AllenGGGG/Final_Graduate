"""Research point 1: the default and ``all`` run the complete workflow."""
import sys


def main():
    if len(sys.argv) == 1 or sys.argv[1] == 'all' or sys.argv[1].startswith('-'):
        from .tools.allocation_final_pipeline import main as execute
        if len(sys.argv) > 1 and sys.argv[1] == 'all':
            del sys.argv[1]
    else:
        from .tools.allocation_final_run import main as execute
    execute()

if __name__ == "__main__":
    main()
