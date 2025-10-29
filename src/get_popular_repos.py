#!/usr/bin/env python3
"""
Fetch the most popular GitHub repositories by language.
"""

import requests
import json
import argparse
from datetime import datetime, timedelta


def get_popular_repos(language="java", count=10, min_stars=1000):
    """
    Fetch popular GitHub repositories using the GitHub API.

    Args:
        language: Programming language to filter by (e.g., "java", "python", "javascript")
        count: Number of repositories to fetch
        min_stars: Minimum number of stars required

    Returns:
        List of repository URLs
    """
    # GitHub API endpoint for searching repositories
    url = "https://api.github.com/search/repositories"

    # Create date for repos created in last 5 years (to get actively maintained ones)
    date_threshold = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    # Query parameters
    params = {
        "q": f"language:{language} stars:>{min_stars} created:>{date_threshold}",
        "sort": "stars",
        "order": "desc",
        "per_page": count
    }

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        repos = []

        for item in data.get("items", []):
            repo_url = item["clone_url"]
            repo_name = item["full_name"]
            stars = item["stargazers_count"]

            print(f"{repo_name} ({stars:,} stars)")
            repos.append(repo_url)

        return repos

    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Get popular GitHub repositories")
    parser.add_argument("--language", default="java", help="Programming language filter")
    parser.add_argument("--count", type=int, default=10, help="Number of repos to fetch")
    parser.add_argument("--min-stars", type=int, default=1000, help="Minimum stars required")
    parser.add_argument("--output", help="Output file to save repo URLs")

    args = parser.parse_args()

    print(f"Fetching top {args.count} {args.language} repositories...")
    repos = get_popular_repos(args.language, args.count, args.min_stars)

    if repos:
        print(f"\nFound {len(repos)} repositories")

        if args.output:
            with open(args.output, "w") as f:
                for repo in repos:
                    f.write(repo + "\n")
            print(f"Saved to {args.output}")
        else:
            print("\nRepository URLs:")
            for repo in repos:
                print(repo)
    else:
        print("Error: No repositories found")


if __name__ == "__main__":
    main()