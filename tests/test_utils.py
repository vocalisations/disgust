import pytest

from disgust.utils import read_feature_set, create_split_videos_masks, get_id_from_video_link
import numpy as np


@pytest.mark.parametrize('feature_string, expected', [
    ('[1.2, 1.0]', np.array([1.2, 1.0])),
    ('NaN', None),
    ('nan', None),
    (np.nan, None),
])
def test_read_feature_set(feature_string, expected):
    result = read_feature_set(feature_string)

    if expected is None:
        assert result is None
    else:
        np.testing.assert_allclose(result, expected, atol=1e-5)


def test_split_videos_without_leakage():
    """Split videos in such a way that similar videos are not in the same set"""
    links = ['https://vm.tiktok.com/ZMYeUb3S3/',
             'https://vm.tiktok.com/ZMYes3H17/',
             'https://www.youtube.com/watch?v=yekWI59YWTg',
             'https://www.youtube.com/watch?v=yekWI59YWTg',
             'https://www.youtube.com/watch?v=ynAUxAwBoPQ',
             'https://www.youtube.com/watch?v=yuDFtmzrwEU',
             'https://www.youtube.com/watch?v=z1jMrStOlQY&list=PLdxpsEsuvjd57hxPXa-vX_sDRp6Lq1OLh&index=5&t=1s',
             'https://www.youtube.com/watch?v=z1jMrStOlQY&list=PLdxpsEsuvjd57hxPXa-vX_sDRp6Lq1OLh&index=5&t=1s',
             'https://www.youtube.com/watch?v=zn4UsUz3yw8&list=PLdxpsEsuvjd57hxPXa-vX_sDRp6Lq1OLh&index=7&t=106s',
             'https://www.youtube.com/watch?v=zn4UsUz3yw8&list=PLdxpsEsuvjd57hxPXa-vX_sDRp6Lq1OLh&index=7&t=106s',
             'https://www.youtube.com/watch?v=zn4UsUz3yw8&list=PLdxpsEsuvjd57hxPXa-vX_sDRp6Lq1OLh&index=7&t=106s',
             'https://www.youtube.com/watch?v=ztaJb8hSmizk',
             'https://youtu.be/296jPy_lJ8o',
             'https://youtu.be/296jPy_lJ8o',
             'https://youtu.be/2mLLSqm72KU',
             'https://youtu.be/2mLLSqm72KU',
             'https://youtu.be/4RmfZtmMhAg',
             'https://youtu.be/4RmfZtmMhAg', ]

    train_masks, validation_masks, test_masks = create_split_videos_masks(links)

    print(sorted(train_masks))
    print(sorted(validation_masks))
    print(sorted(test_masks))

    # Is every link in exactly one split?
    for link_id in [get_id_from_video_link(link) for link in links]:
        in_split = 0
        for split_mask in [train_masks, validation_masks, test_masks]:
            ids_in_split = [get_id_from_video_link(links[i]) for i in split_mask]
            if link_id in ids_in_split:
                in_split += 1
        assert in_split == 1  # id should only be in 1 split


@pytest.mark.parametrize('link, vid', [
    ('https://vm.tiktok.com/ZMYeUb3S3/', 'ttZMYeUb3S3'),
    ('https://vm.tiktok.com/ZMYes3H17/', 'ttZMYes3H17'),
    ('https://www.youtube.com/watch?v=yekWI59YWTg', 'ytyekWI59YWTg'),
    ('https://www.youtube.com/watch?v=ynAUxAwBoPQ', 'ytynAUxAwBoPQ'),
    ('https://www.youtube.com/watch?v=yuDFtmzrwEU', 'ytyuDFtmzrwEU'),
    ('https://www.youtube.com/watch?v=z1jMrStOlQY&list=vX_sDRp6Lq1OLh&index=5&t=1s', 'ytz1jMrStOlQY'),
    ('https://www.youtube.com/watch?v=z1jMrStOlQY&list=vX_sDRp6Lq1OLh&index=5&t=2s', 'ytz1jMrStOlQY'),
    ('https://www.youtube.com/watch?v=zn4UsUz3yw8&list=PLdxpshxPXa-vX_sDRp6Lq1OLh&index=7&t=106s', 'ytzn4UsUz3yw8'),
    ('https://www.youtube.com/watch?v=zn4UsUz3yw8&list=lkasjdlkjasdf-vX_sDRp6Lq1OLh&index=7&t=106s', 'ytzn4UsUz3yw8'),
    ('https://www.youtube.com/watch?v=ztaJb8hSmizk', 'ytztaJb8hSmizk'),
    ('https://youtu.be/296jPy_lJ8o', 'yt296jPy_lJ8o'),
    ('https://youtu.be/2mLLSqm72KU/', 'yt2mLLSqm72KU'),
    ('https://youtu.be/4RmfZtmMhAg', 'yt4RmfZtmMhAg'),
])
def test_get_unique_id_from_link(link, vid):
    assert get_id_from_video_link(link) == vid
