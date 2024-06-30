use std::fs;
use std::path::Path;

pub fn count_files_in_dir<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error>  {
    let entries = fs::read_dir(path)?;
    let mut count = 0;

    for entry in entries {
        let entry = entry?;
        if entry.path().is_file() {
            count += 1;
        }
    }

    Ok(count)
}
