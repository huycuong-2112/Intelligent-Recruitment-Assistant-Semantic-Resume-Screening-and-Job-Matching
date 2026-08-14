import os
import pandas as pd


def build_jd_dataset(
    postings_path,
    industries_path,
    job_industries_path,
    output_path,
    samples_per_industry=500
):
    print("Loading job postings...")

    postings = pd.read_csv(postings_path)

    print(f"Loaded {len(postings):,} job postings.")

    # --------------------------------------------------
    # 1. Load industry mapping
    # --------------------------------------------------
    print("Loading industry data...")

    industries = pd.read_csv(industries_path)
    job_industries = pd.read_csv(job_industries_path)

    # --------------------------------------------------
    # 2. Join job -> industry_id -> industry_name
    # --------------------------------------------------
    job_industries = job_industries.merge(
        industries,
        on="industry_id",
        how="left"
    )

    # --------------------------------------------------
    # 3. Join industry information into postings
    # --------------------------------------------------
    postings = postings.merge(
        job_industries[
            ["job_id", "industry_id", "industry_name"]
        ],
        on="job_id",
        how="left"
    )

    # --------------------------------------------------
    # 4. Remove postings without useful JD text
    # --------------------------------------------------
    postings = postings.dropna(
        subset=["description"]
    )

    postings["description"] = (
        postings["description"]
        .astype(str)
        .str.strip()
    )

    postings = postings[
        postings["description"].str.len() > 100
    ]

    print(
        f"After removing invalid descriptions: "
        f"{len(postings):,}"
    )

    # --------------------------------------------------
    # 5. Keep only records with industry information
    # --------------------------------------------------
    postings = postings.dropna(
        subset=["industry_name"]
    )

    print(
        f"Postings with industry information: "
        f"{len(postings):,}"
    )

    # --------------------------------------------------
    # 6. Remove duplicate job postings
    # --------------------------------------------------
    postings = postings.drop_duplicates(
        subset=["job_id"]
    )

    # --------------------------------------------------
    # 7. Sample evenly across industries
    # --------------------------------------------------
    sampled_groups = []

    for industry, group in postings.groupby(
        "industry_name"
    ):
        sample_size = min(
            samples_per_industry,
            len(group)
        )

        sampled_groups.append(
            group.sample(
                n=sample_size,
                random_state=42
            )
        )

    if not sampled_groups:
        raise ValueError(
            "No valid job postings were found."
        )

    jd_subset = pd.concat(
        sampled_groups,
        ignore_index=True
    )

    # --------------------------------------------------
    # 8. Keep only fields needed for our project
    # --------------------------------------------------
    columns = [
        "job_id",
        "title",
        "description",
        "formatted_experience_level",
        "skills_desc",
        "industry_id",
        "industry_name"
    ]

    jd_subset = jd_subset[
        [
            column
            for column in columns
            if column in jd_subset.columns
        ]
    ]

    # --------------------------------------------------
    # 9. Shuffle final dataset
    # --------------------------------------------------
    jd_subset = jd_subset.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 10. Save
    # --------------------------------------------------
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    jd_subset.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("JD dataset created successfully!")
    print(f"Total JD: {len(jd_subset):,}")
    print(
        f"Industries: "
        f"{jd_subset['industry_name'].nunique():,}"
    )
    print(f"Output: {output_path}")

    print()
    print("Top industries:")
    print(
        jd_subset["industry_name"]
        .value_counts()
        .head(20)
    )


if __name__ == "__main__":

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------
    # Current file:
    # src/Dataloader/jd_dataset_builder.py
    #
    # ../      -> src
    # ../../   -> Project root
    # --------------------------------------------------

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.abspath(
        os.path.join(
            current_dir,
            "..",
            ".."
        )
    )

    postings_path = os.path.join(
        project_root,
        "Data",
        "Raw",
        "JD",
        "Linkedin",
        "postings.csv"
    )

    industries_path = os.path.join(
        project_root,
        "Data",
        "Raw",
        "JD",
        "Linkedin",
        "mappings",
        "industries.csv"
    )

    job_industries_path = os.path.join(
        project_root,
        "Data",
        "Raw",
        "JD",
        "Linkedin",
        "jobs",
        "job_industries.csv"
    )

    output_path = os.path.join(
        project_root,
        "Data",
        "Processed",
        "jd_dataset.csv"
    )

    # --------------------------------------------------
    # Print paths for checking
    # --------------------------------------------------

    print("Project root:")
    print(project_root)
    print()

    print("Postings:")
    print(postings_path)
    print()

    print("Industries:")
    print(industries_path)
    print()

    print("Job industries:")
    print(job_industries_path)
    print()

    print("Output:")
    print(output_path)
    print()

    # --------------------------------------------------
    # Check input files
    # --------------------------------------------------

    if not os.path.exists(postings_path):
        raise FileNotFoundError(
            f"Cannot find postings.csv:\n{postings_path}"
        )

    if not os.path.exists(industries_path):
        raise FileNotFoundError(
            f"Cannot find industries.csv:\n{industries_path}"
        )

    if not os.path.exists(job_industries_path):
        raise FileNotFoundError(
            "Cannot find job_industries.csv:\n"
            f"{job_industries_path}"
        )

    # --------------------------------------------------
    # Build dataset
    # --------------------------------------------------

    build_jd_dataset(
        postings_path,
        industries_path,
        job_industries_path,
        output_path,
        samples_per_industry=500
    )