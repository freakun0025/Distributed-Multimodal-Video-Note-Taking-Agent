import pytest

from common.errors import ErrorCode, PermanentError
from common.youtube_url import canonical_url, parse_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"http://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://WWW.YouTube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=461s",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PL123&index=2",
        f"https://www.youtube.com/watch?feature=share&v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=abc&t=30",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}?start=10",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"youtu.be/{VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
    ],
)
def test_every_supported_url_form_yields_the_video_id(url: str) -> None:
    assert parse_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "https://www.youtube.com/playlist?list=PL123",  # playlist without v=
        "https://www.youtube.com/watch?list=PL123",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQQ",  # 12 chars
        "https://www.youtube.com/watch?v=dQw4w9W$XcQ",
        "https://www.youtube.com/",
        "https://www.youtube.com/@somechannel",
        "https://youtu.be/",
        f"https://vimeo.com/{VIDEO_ID}",
        f"https://www.youtube.com.evil.com/watch?v={VIDEO_ID}",
        f"ftp://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&" + "x" * 2100,
    ],
)
def test_anything_else_is_rejected_as_invalid_url(url: str) -> None:
    with pytest.raises(PermanentError) as caught:
        parse_video_id(url)
    assert caught.value.code is ErrorCode.INVALID_URL


def test_canonical_url_is_the_plain_watch_url() -> None:
    assert canonical_url(VIDEO_ID) == f"https://www.youtube.com/watch?v={VIDEO_ID}"


def test_canonical_url_rejects_a_bad_id() -> None:
    with pytest.raises(PermanentError):
        canonical_url("nope")
