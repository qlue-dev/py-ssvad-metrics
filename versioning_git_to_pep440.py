"""
Convert PEP 440 git tag versioning to PEP 440

Version 2020.11.13
"""
import sys


def versioning_git_to_pep440(version: str) -> str:
    """
    Convert PEP 440 git tag versioning to PEP 440.
    We recommend GitFlow or GitHub Flow,
    but any git workflow works as long as it is properly
    tagged (see examples).

    Please refer to https://www.python.org/dev/peps/pep-0440/ .

    EXAMPLES (git tag = generated package version)
    --------
    * 1.0.0 = 1.0.0
    * 1.0.0a1 = 1.0.0a1
    * 1.0.0b5 = 1.0.0a5
    * 1.0.0.dev3 = 1.0.0.dev3

    Commit since:
    * 1.0.0.dev3-1-ge913d81 = 1.0.0.dev3+1.ge913d81
    * 1.0.0-1-ge913d81 = 1.0.0+1.ge913d81

    With custom tag:
    * 1.0.0-cust = 1.0.0+cust
    * 1.0.0.dev0-cust = 1.0.0.dev0+cust
    * 1.0.0-cust-1-ge913d81 = 1.0.0+cust.1.ge913d81

    NOTE:
    * 1.0.0.dev0 is less than 1.0.0
    * 1.0.0b5 is less than 1.0.0
    * 1.0.0+1.ge913d81 is less than 1.0.0+11.39fa12ec
    """
    tk_ver = version.split("-")
    if len(tk_ver) <= 1:
        return version
    return tk_ver[0] + "+" + ".".join(tk_ver[1:])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(versioning_git_to_pep440(sys.argv[1]))
    else:
        print(versioning_git_to_pep440(sys.stdin.read().splitlines()[0]))
