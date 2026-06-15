"""
store.py  — backend router
==========================
Switches between the SQLite store (store_sqlite.py) and the original
JSON-file store (store_json.py) based on config.USE_DATABASE.

ROLLBACK: set USE_DATABASE = False in config.py -> instantly reverts
to the JSON backend with no other code changes required.
"""

import config

if config.USE_DATABASE:
    from store_sqlite import *          # noqa: F401, F403
    from store_sqlite import (          # explicit re-export for IDE / type checkers
        APPROVED_STATUSES, DENIED_STATUSES, SEED_USERS, now_iso,
        load_users, save_users, save_user,
        find_user_by_email, find_user_by_id, update_user,
        load_uploads, save_uploads, add_upload,
        uploads_for_user, find_upload_by_id, delete_upload,
        load_requests, save_requests, add_request,
        set_request_status, requests_for_owner,
        requests_by_customer, count_denials, approved_owner_ids,
    )
else:
    from store_json import *            # noqa: F401, F403
    from store_json import (            # explicit re-export
        APPROVED_STATUSES, DENIED_STATUSES, SEED_USERS, now_iso,
        load_users, save_users,
        find_user_by_email, find_user_by_id, update_user,
        load_uploads, save_uploads, add_upload,
        uploads_for_user, find_upload_by_id, delete_upload,
        load_requests, save_requests, add_request,
        set_request_status, requests_for_owner,
        requests_by_customer, count_denials, approved_owner_ids,
    )
