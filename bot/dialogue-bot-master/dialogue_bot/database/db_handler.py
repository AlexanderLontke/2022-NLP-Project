import logging
import os
import re
import shutil
from typing import Iterable, List, Tuple

import nltk
from dialogue_bot.utils.nlp import word_variations, stopwords
from whoosh.index import open_dir, create_in
from whoosh.query import Or, Term

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from pymongo import MongoClient, errors

MONGO_URL = os.getenv("MONGO_URL", "0.0.0.0:27017")
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "vwl-chatbot")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "monopolytask")


def limit_it(it, limit):
    for i, x in enumerate(it):
        if (limit is not None) and (i == limit):
            break
        yield x


class DBHandler(object):
    def __init__(self, bot_id):
        self.db_name = self.fix_db_name("dialogue-bot2-{}".format(bot_id))
        self.client = None
        self.db = None
        self.connect()

    @staticmethod
    def fix_db_name(db_name: str) -> str:
        return re.sub("[^0-9a-zA-Z-]", "-", db_name)

    def connect(self):
        host = MONGO_URL
        logger.info("Try to connect to MongoDB instance at {}...".format(host))

        # try to instantiate a client instance
        try:
            self.client = MongoClient(
                host=[host],
                serverSelectionTimeoutMS=10000,  # 10 second timeout
                username=MONGO_USERNAME,
                password=MONGO_PASSWORD,
            )
            self.db = self.client[self.db_name]
            logger.info("Successfully established connection to MongoDB instance")
            logger.info(
                "\tMongoDB server version: {}".format(
                    self.client.server_info()["version"]
                )
            )
            logger.info(
                "\tMongoDB databases:      {}".format(self.client.list_database_names())
            )
            logger.info(
                "\tMongoDB collections:    {}".format(self.db.list_collection_names())
            )

        except errors.ServerSelectionTimeoutError as err:
            self.client = None
            self.db = None
            logger.error("Failed establishing a connection to MongoDB")
            logger.error(err)

    def exists_collection(self, collection_name: str) -> bool:
        return collection_name in self.db.list_collection_names()

    def delete_collection(self, collection_name: str):
        if self.exists_collection(collection_name):
            logger.info('Delete collection "{}"'.format(collection_name))
            self.db.drop_collection(collection_name)

    def ensure_collection(self, collection_name: str):
        logger.info('Ensure collection "{}"'.format(collection_name))
        pass

    def index_object(self, collection_name: str, obj: dict, key_attributes: set):
        """
        If key_attributes is None, this will insert a new DB entry without checking for duplicates.
        If  key_attributes is set, then an already existing entry will be overwritten with the new object
        """

        if (key_attributes is None) or (len(key_attributes) <= 0):
            # This will index objects with duplicates
            self.db[collection_name].insert_one(obj)
        else:
            # This will index objects without duplications
            query = {k: obj[k] for k in key_attributes}
            self.db[collection_name].replace_one(query, obj, upsert=True)

    def commit(self, collection_name: str):
        """Supposed to be called after indexing objects"""
        pass

    def find(self, collection_name: str, query=None, limit=None) -> Iterable:
        """
        Returns all objects matching a specific query.
        If query is None, then all objects are returned.
        """
        if query is None:
            it = self.db[collection_name].find()
        else:
            it = self.db[collection_name].find(query)
        if limit is not None:
            it = it.limit(limit)
        for x in it:
            yield x

    def find_one(self, *args, **kwargs):
        for x in self.find(*args, limit=1, **kwargs):
            return x
        return None

    def query_attr_in_value_list(self, attr: str, lst) -> dict:
        """Constructs a query which resolves to True, if an attribute value is part of a list of values"""
        lst = list(set(lst))
        return {attr: {"$in": lst}}


class IndexHandler(object):
    def __init__(self, root_index_dirpath: str):
        self._root_index_dirpath = root_index_dirpath
        self._index_data = {}

    def _index_directory(self, index_name: str) -> str:
        return os.path.join(self._root_index_dirpath, index_name)

    def exists_index(self, index_name: str) -> bool:
        index_dir = self._index_directory(index_name)
        return os.path.exists(index_dir)

    def delete_index(self, index_name: str):
        index_dir = self._index_directory(index_name)
        if os.path.exists(index_dir):
            logger.info('Delete index at "{}"'.format(index_dir))
            shutil.rmtree(index_dir)
        if index_name in self._index_data:
            del self._index_data[index_name]

    def ensure_index(self, index_name: str, schema):
        index_dir = self._index_directory(index_name)
        if self.exists_index(index_name):
            logger.info('Open index at "{}"'.format(index_dir))
            idx = open_dir(index_dir)
        else:
            logger.info('Create index at "{}"'.format(index_dir))
            os.makedirs(index_dir)
            idx = create_in(index_dir, schema)
        self._index_data[index_name] = {"idx": idx, "writer": None}

    def index_object(self, index_name: str, obj: dict):
        index_data = self._index_data[index_name]
        if index_data["writer"] is None:
            # open writer
            index_data["writer"] = index_data["idx"].writer()
        index_data["writer"].add_document(**obj)

    def commit(self, index_name: str):
        index_data = self._index_data[index_name]
        if index_data["writer"] is not None:
            # commit & close writer
            index_data["writer"].commit()
            index_data["writer"] = None

    def find(
        self, index_name: str, get_query: dict = None, search_query=None, limit=None
    ):
        """
        Returns all objects matching a specific query.
        If query is None, then all objects are returned.
        """
        index_data = self._index_data[index_name]
        if (get_query is None) and (search_query is None):
            with index_data["idx"].searcher() as searcher:
                for d in limit_it(searcher.documents(), limit):
                    yield d
        elif get_query is not None:
            with index_data["idx"].searcher() as searcher:
                for d in limit_it(searcher.documents(**get_query), limit):
                    yield d
        elif search_query is not None:
            with index_data["idx"].searcher() as searcher:
                for d in searcher.search(search_query, limit=limit):
                    yield {"score": d.score, "obj": dict(d), "hit": d}
        else:
            raise AssertionError("Please choose either get_query or search_query")

    def find_one(self, *args, **kwargs):
        for x in self.find(*args, limit=1, **kwargs):
            return x
        return None

    def extract_keywords(
        self,
        index_name: str,
        lang: str,
        fieldname: str,
        text: str,
        ignore_stopwords=False,
    ) -> List[Tuple[str, float]]:
        index_data = self._index_data[index_name]
        keywords = []
        with index_data["idx"].searcher() as searcher:
            try:
                keywords = searcher.key_terms_from_text(fieldname, text)
            except:
                pass
        if len(keywords) <= 0:
            logger.error(
                'Failed to extract keywords from "{}". Using fallback method.'.format(
                    text
                )
            )
            keywords = [(w, 1.0) for w in nltk.word_tokenize(text)]
            if ignore_stopwords:
                keywords = [
                    (w, s) for w, s in keywords if w.lower() not in set(stopwords(lang))
                ]
        logger.debug("WhooshIndex: Extracted keywords: {}".format(keywords))
        return keywords

    def query_attr_in_value_list(self, attr: str, lst):
        """Constructs a query which resolves to True, if an attribute value is part of a list of values"""
        query = Or([Term(attr, x) for x in lst])
        return query

    def more_like_this_query(
        self, index_name: str, lang: str, fieldname: str, text: str, use_variations=True
    ):
        """This can be used as a search_query for IndexHandler.find(...)"""
        keywords = self.extract_keywords(
            index_name, lang, fieldname, text, ignore_stopwords=False
        )

        if use_variations:
            _keywords = []
            for word, weight in keywords:
                _keywords.extend([(w, weight) for w in word_variations(lang, word)])
            keywords = _keywords[:]

        query = Or([Term(fieldname, word, boost=weight) for word, weight in keywords])
        return query


if __name__ == "__main__":
    import settings
    from whoosh.fields import Schema, ID, TEXT

    pass

    # h = DBHandler('test')
    # h.index_object('test', {'name': 'Christina', 'age': 32}, {'name'})
    # print(h.find_one('test'))

    schema = Schema(name=ID(stored=True), text=TEXT(stored=True))
    h = IndexHandler(
        os.path.join(settings.PROJECT_DIRPATH, "data", "generated", "deleteme")
    )
    h.ensure_index("test", schema)
    # h.index_object('test', {'name': 'Paul', 'text': 'I am Paul'})
    # h.commit('test')

    for x in h.find("test"):
        print(x)
