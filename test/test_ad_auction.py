def test_ad_auction():
    from ad_auction import AdAuction

    aa = AdAuction(config_file_name="test/config-test.json")

    assert abs(aa.run_episode() - 7.3771870) < 1e-4
