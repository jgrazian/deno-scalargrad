/** Loads number data from csv file.
* @param header Set to true to ignore first line of file
*/
export function loadCsv(path: string, header = true): number[][] {
    return Deno.readTextFileSync(path)
        .trim()
        .split('\n')
        .slice((header) ? 1 : 0)
        .map(l => l.split(',')
            .map(v => Number(v.trim())
            )
        );
}

/** Create random permutation of array [0..n-1] */
export function randomPermutation(n: number): number[] {
    const array = [...Array(n).keys()];
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}
