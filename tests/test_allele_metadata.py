"""Regression coverage for operation-local allele metadata reuse."""
import unittest
from unittest.mock import patch

import pandas as pd
import pypgx
from pypgx.api import core
from pypgx.sdk import utils as sdk_utils


def reference_collapse(gene, alleles, assembly):
    """Original implementation, using the public uncached metadata APIs."""
    results = []
    for a in alleles:
        result = False
        for b in alleles:
            if a == b:
                continue
            v1 = set(core.list_variants(gene, alleles=a, assembly=assembly, mode='core'))
            v2 = set(core.list_variants(gene, alleles=b, assembly=assembly, mode='core'))
            if v1.issubset(v2):
                result = True
                break
        results.append(result)
    return [x for i, x in enumerate(alleles) if not results[i]]


def reference_priority(gene, allele, assembly):
    a = core.FUNCTION_ORDER.index(core.get_function(gene, allele))
    variants = core.list_variants(gene, alleles=allele, assembly=assembly, mode='core')
    impacts = [core.get_variant_impact(x) for x in variants]
    return (a, -len(variants), -len([x for x in impacts if x]),
            allele == core.get_ref_allele(gene))


class TestAlleleMetadata(unittest.TestCase):
    def test_collapse_against_original(self):
        cases = {
            'CYP2B6': ['*1', '*4', '*6', '*7', '*9', '*6'],
            'CYP2D6': ['*1', '*2', '*4', '*5', '*13', '*36'],
            'CYP4F2': ['*1', '*2', '*3'],
            'UGT1A1': ['*1', '*28', '*36', '*37', '*80+*37'],
        }
        for assembly in ['GRCh37', 'GRCh38']:
            for gene, alleles in cases.items():
                with self.subTest(gene=gene, assembly=assembly):
                    self.assertEqual(
                        reference_collapse(gene, alleles, assembly),
                        core.collapse_alleles(gene, alleles, assembly))

    def test_priority_against_original(self):
        for assembly in ['GRCh37', 'GRCh38']:
            for gene in ['CYP2B6', 'CYP2D6', 'DPYD', 'UGT1A1', 'CYP4F2']:
                df = core.load_allele_table()
                df = df[(df.Gene == gene) & df.Function.notna()]
                alleles = df.StarAllele.to_list()[::max(1, len(df) // 12)]
                alleles += alleles[:2]
                alleles.reverse()
                with self.subTest(gene=gene, assembly=assembly):
                    self.assertEqual(
                        sorted(alleles, key=lambda a: reference_priority(gene, a, assembly)),
                        core.sort_alleles(alleles, by='priority', gene=gene, assembly=assembly))

    def test_lazy_validation(self):
        self.assertEqual([], core.collapse_alleles('invalid', [], 'invalid'))
        self.assertEqual(['bad'], core.collapse_alleles('invalid', ['bad'], 'invalid'))
        self.assertEqual(['bad', 'bad'], core.collapse_alleles('invalid', ['bad', 'bad']))
        self.assertEqual([], core.sort_alleles([], by='priority'))
        with self.assertRaisesRegex(ValueError, 'Gene is required'):
            core.sort_alleles(['*1'], by='priority')
        with self.assertRaises(sdk_utils.NotTargetGeneError):
            core.collapse_alleles('invalid', ['*1', '*2'])
        with self.assertRaises(sdk_utils.AlleleNotFoundError):
            core.collapse_alleles('CYP2B6', ['*1', '*bad'])
        with self.assertRaises(KeyError):
            core.sort_alleles(['*1'], by='priority', gene='CYP2B6', assembly='invalid')
        table = core.load_allele_table()
        table.loc[(table.Gene == 'CYP2D6') & (table.StarAllele == '*1'), 'Function'] = float('nan')
        with patch.object(core, 'load_allele_table', return_value=table):
            with self.assertRaises(ValueError):
                core.sort_alleles(['*1'], by='priority', gene='CYP2D6')

    def test_equal_definitions_and_stable_priority_ties(self):
        table = core.load_allele_table()
        first = (table.Gene == 'CYP4F2') & (table.StarAllele == '*2')
        second = (table.Gene == 'CYP4F2') & (table.StarAllele == '*3')
        for assembly in ['GRCh37', 'GRCh38']:
            table.loc[second, f'{assembly}Core'] = table.loc[first, f'{assembly}Core'].values[0]
        table.loc[second, 'Function'] = table.loc[first, 'Function'].values[0]
        with patch.object(core, 'load_allele_table', return_value=table):
            for assembly in ['GRCh37', 'GRCh38']:
                self.assertEqual([], core.collapse_alleles('CYP4F2', ['*2', '*3'], assembly))
                self.assertEqual(['*3', '*2', '*3'], core.sort_alleles(
                    ['*3', '*2', '*3'], by='priority', gene='CYP4F2', assembly=assembly))

    def test_independent_public_tables(self):
        for loader in [core.load_gene_table, core.load_allele_table, core.load_variant_table]:
            with self.subTest(loader=loader.__name__):
                original = loader()
                changed = loader()
                changed.iloc[0, 0] = 'modified'
                pd.testing.assert_frame_equal(original, loader())

    def test_table_reads_are_bounded_within_prediction(self):
        archive = pypgx.Archive.from_file('test-data/CYP4F2-GRCh37.zip')
        original = core.load_allele_table
        with patch.object(core, 'load_allele_table', wraps=original) as loaded:
            first = pypgx.predict_alleles(archive)
            reads = loaded.call_count
        sample = archive.data.samples[0]
        for i in range(12):
            archive.data.df[f'copy{i}'] = archive.data.df[sample]
        with patch.object(core, 'load_allele_table', wraps=original) as loaded:
            repeated = pypgx.predict_alleles(archive)
            self.assertEqual(reads, loaded.call_count)
        pd.testing.assert_series_equal(first.data.loc[sample], repeated.data.loc[sample])

    def test_missing_calls(self):
        for assembly in ['GRCh37', 'GRCh38']:
            archive = pypgx.Archive.from_file(f'test-data/CYP4F2-{assembly}.zip')
            for sample in archive.data.samples:
                archive.data.df[sample] = '.|.:.'
            result = pypgx.predict_alleles(archive)
            self.assertTrue((result.data.Haplotype1 == '*1;').all())
            self.assertTrue((result.data.Haplotype2 == '*1;').all())
            self.assertTrue((result.data.AlternativePhase == ';').all())


if __name__ == '__main__':
    unittest.main()
