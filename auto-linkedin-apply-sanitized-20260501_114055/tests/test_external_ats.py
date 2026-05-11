from modules.external_ats import (
    ExternalApplyJob,
    external_apply_review_rows,
    classify_ats_platform,
    external_url_needs_recapture,
    extract_real_external_apply_url,
    is_real_external_apply_url,
    load_external_apply_jobs,
    platform_label,
    summarize_platforms,
)


def test_is_real_external_apply_url_rejects_linkedin_urls():
    linkedin_job_url = "https://www.linkedin.com/jobs/view/123"

    assert not is_real_external_apply_url(linkedin_job_url, linkedin_job_url)
    assert not is_real_external_apply_url(
        "https://www.linkedin.com/jobs/search/?currentJobId=123", linkedin_job_url
    )
    assert not is_real_external_apply_url("not a url", linkedin_job_url)


def test_is_real_external_apply_url_accepts_non_linkedin_ats_url():
    assert is_real_external_apply_url(
        "https://jobs.ashbyhq.com/example/abc123",
        "https://www.linkedin.com/jobs/view/123",
    )


def test_extract_real_external_apply_url_unwraps_linkedin_redirect():
    redirect_url = (
        "https://www.linkedin.com/safety/go?"
        "url=https%3A%2F%2Fboards.greenhouse.io%2Facme%2Fjobs%2F1"
    )

    assert (
        extract_real_external_apply_url(
            redirect_url, "https://www.linkedin.com/jobs/view/123"
        )
        == "https://boards.greenhouse.io/acme/jobs/1"
    )
    assert classify_ats_platform(redirect_url) == "greenhouse"


def test_classify_major_ats_platforms():
    assert (
        classify_ats_platform("https://boards.greenhouse.io/acme/jobs/1")
        == "greenhouse"
    )
    assert classify_ats_platform("https://jobs.lever.co/acme/1") == "lever"
    assert classify_ats_platform("https://jobs.ashbyhq.com/acme/1") == "ashby"
    assert classify_ats_platform("https://jobs.workable.com/view/abc") == "workable"
    assert (
        classify_ats_platform("https://jobs.smartrecruiters.com/acme/1")
        == "smartrecruiters"
    )
    assert classify_ats_platform("https://acme.breezy.hr/p/123") == "breezy"
    assert (
        classify_ats_platform("https://acme.wd1.myworkdaysite.com/recruiting/job/1")
        == "workday"
    )
    assert classify_ats_platform("https://acme.icims.com/jobs/1/job") == "icims"
    assert classify_ats_platform("https://recruiting2.ultipro.com/acme") == "ukg"
    assert (
        classify_ats_platform("https://example.com/careers?gh_jid=4623595007")
        == "greenhouse"
    )
    assert (
        classify_ats_platform("https://example.com/about?ashby_jid=abc123") == "ashby"
    )


def test_classify_observed_custom_career_sites():
    assert classify_ats_platform("https://www.amazon.jobs/en/jobs/123") == "amazon_jobs"
    assert (
        classify_ats_platform("https://jobs.careers.microsoft.com/us/en/job/123")
        == "microsoft_careers"
    )
    assert (
        classify_ats_platform("https://lifeattiktok.com/search/123") == "tiktok_careers"
    )
    assert (
        classify_ats_platform("https://apply.deloitte.com/careers/InviteToApply")
        == "deloitte_careers"
    )
    assert classify_ats_platform("https://careers.ey.com/ey/job/123") == "ey_careers"
    assert classify_ats_platform("https://wellfound.com/jobs/123") == "wellfound"
    assert (
        classify_ats_platform("https://www.cybercoders.com/job-detail/123")
        == "cybercoders"
    )
    assert classify_ats_platform("https://recruitcrm.io/apply/123") == "recruitcrm"
    assert classify_ats_platform("https://www.remotehunter.com/apply-with-ai/123") == (
        "remotehunter"
    )
    assert classify_ats_platform("https://7seventy.net/job/123") == "7seventy"


def test_external_url_needs_recapture_for_linkedin_and_invalid_urls():
    assert external_url_needs_recapture("https://www.linkedin.com/jobs/view/123")
    assert external_url_needs_recapture("")
    assert not external_url_needs_recapture("https://jobs.lever.co/acme/1")


def test_load_external_apply_jobs_and_summarize_platforms(tmp_path):
    backlog = tmp_path / "manual_writeup_jobs.csv"
    backlog.write_text(
        "\n".join(
            [
                "Job ID,Title,Company,Backlog Type,Job Link,Application Link,Date Saved",
                (
                    "1,Engineer,Acme,external_apply,https://www.linkedin.com/jobs/view/1,"
                    "https://jobs.lever.co/acme/1,2026-04-29"
                ),
                (
                    "2,Researcher,Lab,easy_apply_manual_writeup,"
                    "https://www.linkedin.com/jobs/view/2,Easy Apply,2026-04-29"
                ),
                (
                    "3,Builder,Widgets,external_apply,https://www.linkedin.com/jobs/view/3,"
                    "https://www.linkedin.com/jobs/search/?currentJobId=3,2026-04-29"
                ),
            ]
        ),
        encoding="utf-8",
    )

    jobs = load_external_apply_jobs(backlog)

    assert [job.job_id for job in jobs] == ["1", "3"]
    assert jobs[0].platform == "lever"
    assert jobs[1].platform == "linkedin"
    assert platform_label(jobs[0].platform) == "Lever"
    assert summarize_platforms(jobs) == {"lever": 1, "linkedin": 1}


def test_external_apply_review_rows_show_recapture_and_resolved_url():
    jobs = [
        ExternalApplyJob(
            job_id="1",
            title="Engineer",
            company="Acme",
            job_link="https://www.linkedin.com/jobs/view/1",
            application_link="https://jobs.lever.co/acme/1",
            platform="lever",
            date_saved="2026-04-29",
        ),
        ExternalApplyJob(
            job_id="2",
            title="Builder",
            company="Widgets",
            job_link="https://www.linkedin.com/jobs/view/2",
            application_link="https://www.linkedin.com/jobs/search/?currentJobId=2",
            platform="linkedin",
            date_saved="2026-04-29",
        ),
    ]

    rows = external_apply_review_rows(jobs)

    assert rows[0]["Platform"] == "Lever"
    assert rows[0]["Needs Recapture"] == "no"
    assert rows[0]["Resolved ATS URL"] == "https://jobs.lever.co/acme/1"
    assert rows[1]["Platform"] == "LinkedIn"
    assert rows[1]["Needs Recapture"] == "yes"
    assert rows[1]["Resolved ATS URL"] == ""
